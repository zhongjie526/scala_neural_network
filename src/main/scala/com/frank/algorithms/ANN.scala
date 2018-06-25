package com.frank.algorithms
import com.frank._
import scala.util.control.Breaks._
import breeze.linalg._
import scala.collection.mutable.ArrayBuffer

case class ANN(architecture:List[Int],actFunc:Double=>Double) {
  
  override def toString = s"Neural Network with arhitecture ${architecture.map(_.toString).reduce(_+" ==> "+_)}"
  
  private val rand = new scala.util.Random
  private val n_levels = architecture.size
  private val n_units= scala.collection.mutable.ArrayBuffer.empty[Int]
  
  private val zs = ArrayBuffer.empty[DenseVector[Double]]
  private val as = ArrayBuffer.empty[DenseVector[Double]]
  private val deltas = ArrayBuffer.empty[DenseVector[Double]]
  private val ds = ArrayBuffer.empty[DenseMatrix[Double]]
  
  var printCost = true
  
  def initialize :List[DenseMatrix[Double]] ={
    val thetas = scala.collection.mutable.ArrayBuffer.empty[DenseMatrix[Double]]
    for (l <- 0 until n_levels-1) {
       thetas += DenseMatrix.rand(architecture(l+1),architecture(l)+1).map(_*2-1)
    }
    thetas.toList
  }
  
  def reset_gradient: Unit ={
    this.ds.clear
    for (l <- 0 until n_levels-1) {
       this.ds += DenseMatrix.zeros[Double](architecture(l+1),architecture(l)+1)
    }
  }
  
  
  def get_ds: List[DenseMatrix[Double]] = this.ds.toList
  
  def unrolling_thetas(thetas:List[DenseMatrix[Double]]): Array[Double] ={
    val thetas_unrolled = ArrayBuffer.empty[Double]
    thetas.foreach(T => thetas_unrolled ++= T.toArray)
    thetas_unrolled.toArray
  }
  
  
  def rolling_thetas(thetas_unrolled:Array[Double]): List[DenseMatrix[Double]] ={
    val thetas = ArrayBuffer.empty[DenseMatrix[Double]]
    
    def roll(n_level_rolled:Int,thetas_unrolled:Array[Double]): Unit = {
      if(n_level_rolled >= n_levels) {}
      else {
        val n_rows = architecture(n_level_rolled)
        val n_cols = architecture(n_level_rolled-1)+1
        val n_rolling = n_rows * n_cols
        thetas += DenseMatrix(thetas_unrolled.take(n_rolling)).reshape(n_rows,n_cols)
        roll(n_level_rolled+1,thetas_unrolled.drop(n_rolling))
      }
    }
    
    roll(1,thetas_unrolled)
    thetas.toList
  }
  
  def accumulate_Ds: Unit ={
    for(i <- 0 until deltas.size) {      
      ds(i) += deltas(i) * as(i).t 
    }
  }
  
  def forwardProp(x:DenseVector[Double],thetas: List[DenseMatrix[Double]]) : DenseVector[Double] = { 
    zs.clear
    as.clear
    
    if(x.size != this.architecture.head) 
    {
      println("size of input does Not fit the architecture of the neural network")
      println(s"${this.architecture.head} input nodes expected but ${x.size} observed")
      break
    }

    def getOutput(input: DenseVector[Double],thetas: List[DenseMatrix[Double]]) : DenseVector[Double]={
      val activation = DenseVector(1.0 +: input.map(actFunc).toArray)
      as += activation
      if(thetas.isEmpty) input
      else {
        val z = thetas.head * activation
        zs += z
        getOutput(z,thetas.tail)
      }
    }
    zs += x
    getOutput(x,thetas.toList)
  }
  
  def forwardProp(X: Array[DenseVector[Double]],thetas: List[DenseMatrix[Double]]) : Array[DenseVector[Double]] = X.map(forwardProp(_,thetas)) 

  def backProp(y_hat:DenseVector[Double],y:DenseVector[Double],thetas:List[DenseMatrix[Double]]): Unit ={
    deltas.clear
    
    if(y.size != this.architecture.last) {println("size of output does Not fit the architecture of the neural network");break}
    if(y.size != y_hat.size) {println("size of output does NOT match the size of the predicted output");break}
    
    def getDelta(delta:DenseVector[Double],thetas:List[DenseMatrix[Double]], as:List[DenseVector[Double]]) :Unit ={
      
      if(thetas.isEmpty) {}
      else{
        delta +=: deltas
        val del = (thetas.last.t * delta) :* as.last :* (as.last.map(1.0-_)) 
        //val del = (thetas.last.t * delta)
        getDelta(DenseVector(del.toArray.tail), thetas.init, as.init)
      }
    }
    getDelta(y_hat-y,thetas,this.as.toList.init)
  }
  
  def get_gradient(thetas: List[DenseMatrix[Double]],x:Array[DenseVector[Double]], y:Array[DenseVector[Double]],lambda_reg:Double) :Unit ={
    reset_gradient
    if(x.size != y.size) {println("size of X does NOT match that of y");break}
    else {
        for(i<- 0 until x.size) {
          this.backProp(this.forwardProp(x(i),thetas), y(i),thetas)
          this.accumulate_Ds
        }
        
        for(i <- 0 until ds.size) {
          ds(i) = (ds(i) + thetas(i).map(_*lambda_reg)).map(_/x.size.toDouble)
        }
    }
  }
  
  def update_thetas(thetas: List[DenseMatrix[Double]],lambda:Double) :List[DenseMatrix[Double]] ={
    
    thetas.zip(this.ds).map{case (t,d)=> t-d.map(_*lambda)}
  }
  
  def optimize_gradient(thetas: List[DenseMatrix[Double]],x:Array[DenseVector[Double]], y:Array[DenseVector[Double]],lambda:Double,lambda_reg:Double,n_iter:Int) : List[DenseMatrix[Double]] ={
    def update_gradient(thetas: List[DenseMatrix[Double]],x:Array[DenseVector[Double]], y:Array[DenseVector[Double]],lambda:Double,lambda_reg:Double,iter:Int) : List[DenseMatrix[Double]] ={
      if(printCost) {
        println(s"======================iteration $iter===========================")
        println(s"cost is ${get_cost(x,y,thetas)}")
      }
      
      if(iter <= n_iter){
        
        get_gradient(thetas,x,y,lambda_reg)
        update_gradient( update_thetas(thetas,lambda)  ,x,y,lambda,lambda_reg,iter+1)
      }
      else thetas

    }
    update_gradient(thetas,x,y,lambda,lambda_reg,1)
    
  }
  
  def get_rsquare(x:Array[DenseVector[Double]], y:Array[DenseVector[Double]],thetas:List[DenseMatrix[Double]]) :DenseVector[Double]={
    val y_h = this.forwardProp(x, thetas)
    val sse = y_h.zip(y).map{case (y_h,y) => (y_h-y)}.reduce{(a,b)=> a + (b :^2.0)}
    
    val y_mean = y.reduce(_+_)/y.size.toDouble
    val sst = y.map(v=>(v-y_mean)).reduce{(a,b)=> a + (b :^ 2.0)}
    (sst-sse)/sst
  }
  
  
  def get_cost(X:Array[DenseVector[Double]], y:Array[DenseVector[Double]],thetas: List[DenseMatrix[Double]]) :Double = {
    if(X.size != y.size) {println("size of X does NOT match that of y");break}
    else{
      var cost:Double = 0
      for(i <- 0 until X.size) {
        cost += (this.forwardProp(X(i),thetas)-y(i)).sumOfSquares
      }
      cost/2/X.size
    }
  }
  
  def get_cost_regularized(X:Array[DenseVector[Double]], y:Array[DenseVector[Double]],thetas: List[DenseMatrix[Double]],lambda:Double) :Double = {
    if(X.size != y.size) {println("size of X does NOT match that of y");break}
    else{
      var cost:Double = 0
      cost += X.zip(y).map{case (x,y)=> (this.forwardProp(x,thetas)-y).sumOfSquares}.sum
      cost += this.unrolling_thetas(thetas).map(Math.pow(_,2)).sum * lambda
      cost/2/X.size
    }
  }
  
  def gradient_checking(X:Array[DenseVector[Double]], y:Array[DenseVector[Double]],thetas: List[DenseMatrix[Double]],epsilong:Double,lambda:Double) : Unit={
    val gradient = ArrayBuffer.empty[Double]
    val thetas_unrolled = this.unrolling_thetas(thetas)
    this.get_gradient(thetas,X, y,lambda)
    
    for(i <- 0 until thetas_unrolled.size) {
      val thetas_unrolled_plus = thetas_unrolled.clone
      val thetas_unrolled_minus = thetas_unrolled.clone
      thetas_unrolled_plus(i) += epsilong
      thetas_unrolled_minus(i) -= epsilong
      val thetas_plus = this.rolling_thetas(thetas_unrolled_plus)
      val thetas_minus = this.rolling_thetas(thetas_unrolled_minus)
      val cost_plus = this.get_cost_regularized(X, y, thetas_plus,lambda)
      val cost_minus = this.get_cost_regularized(X, y, thetas_minus,lambda)
      gradient += (cost_plus - cost_minus)/2/epsilong
    }
    
    (gradient.toArray zip this.unrolling_thetas(this.get_ds)).foreach(println)

  }



    
}

