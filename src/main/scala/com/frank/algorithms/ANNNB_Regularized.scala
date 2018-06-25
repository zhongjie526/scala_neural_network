package com.frank.algorithms
import com.frank._
import scala.util.control.Breaks._
import breeze.linalg._
import scala.collection.mutable.ArrayBuffer

case class ANNNB_regularized(architecture:List[Int],actFunc:Double=>Double) {
  
  override def toString = s"Neural Network with arhitecture ${architecture.map(_.toString).reduce(_+" ==> "+_)}"
  
  private val rand = new scala.util.Random
  private val n_levels = architecture.size
  private val n_units= scala.collection.mutable.ArrayBuffer.empty[Int]
  private val thetas = scala.collection.mutable.ArrayBuffer.empty[DenseMatrix[Double]]
  
  private val zs = ArrayBuffer.empty[DenseVector[Double]]
  private val as = ArrayBuffer.empty[DenseVector[Double]]
  private val deltas = ArrayBuffer.empty[DenseVector[Double]]
  private val ds = ArrayBuffer.empty[DenseMatrix[Double]]
  
  def initialize :Unit ={
    for (l <- 0 until n_levels-1) {
       this.thetas += DenseMatrix.rand(architecture(l+1),architecture(l)).map(_*2-1)
       this.ds += DenseMatrix.zeros[Double](architecture(l+1),architecture(l))
    }
  }
  
  def reset_gradient: Unit ={
    this.ds.clear
    for (l <- 0 until n_levels-1) {
       this.ds += DenseMatrix.zeros[Double](architecture(l+1),architecture(l))
    }
  }
  
  def get_thetas: List[DenseMatrix[Double]] = this.thetas.toList
  
  def get_ds: List[DenseMatrix[Double]] = this.ds.toList
  
  def unrolling_thetas(thetas:List[DenseMatrix[Double]]): Array[Double] ={
    val thetas_unrolled = ArrayBuffer.empty[Double]
    thetas.foreach(T => thetas_unrolled ++= T.toArray)
    thetas_unrolled.toArray
  }
  
  def set_thetas(thetas_unrolled:Array[Double]): Unit = {
    this.thetas.clear
    this.thetas ++= this.rolling_thetas(thetas_unrolled)
  }
  
  def get_thetas_unrolled: Array[Double] = this.unrolling_thetas(this.get_thetas)
  
  def rolling_thetas(thetas_unrolled:Array[Double]): List[DenseMatrix[Double]] ={
    val thetas = ArrayBuffer.empty[DenseMatrix[Double]]
    
    def roll(n_level_rolled:Int,thetas_unrolled:Array[Double]): Unit = {
      if(n_level_rolled >= n_levels) {}
      else {
        val n_rows = architecture(n_level_rolled)
        val n_cols = architecture(n_level_rolled-1)
        val n_rolling = n_rows * n_cols
//        thetas_unrolled.take(n_rolling).foreach(println)
        thetas += DenseMatrix(thetas_unrolled.take(n_rolling)).reshape(n_rows,n_cols)
        roll(n_level_rolled+1,thetas_unrolled.drop(n_rolling))
      }
    }
    
    roll(1,thetas_unrolled)
    thetas.toList
  }
  
  def print_n_levels :Unit = println(s"There are ${this.n_levels} levels in the network")
  
  def print_thetas :Unit = for(i <- 1 to this.thetas.size) println(s"Theta $i = \n ${this.thetas(i-1)}")
  
  def print_ds: Unit = for(i <- 1 to ds.size) println(s"Delta${i} = \n ${this.ds(i-1)}")
  
  def print_zs: Unit = for(i <- 1 to zs.size) println(s"z${i} = ${zs(i-1)}")
  
  def print_as: Unit = for(i <- 1 to as.size) println(s"a${i} = ${as(i-1)}")
  
  def print_deltas: Unit = for(i <- 1 to deltas.size) println(s"delta${i+1} = ${this.deltas(i-1)}")
  
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

    def getOutput(spike: DenseVector[Double],thetas: List[DenseMatrix[Double]]) : DenseVector[Double]={
      val activation = spike.map(actFunc)
      as += activation
      if(thetas.isEmpty) spike
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

  def backProp(y_hat:DenseVector[Double],y:DenseVector[Double]): Unit ={
    deltas.clear
    
    if(y.size != this.architecture.last) {println("size of output does Not fit the architecture of the neural network");break}
    if(y.size != y_hat.size) {println("size of output does NOT match the size of the predicted output");break}
    
    def getDelta(delta:DenseVector[Double],thetas:List[DenseMatrix[Double]], as:List[DenseVector[Double]]) :Unit ={
      
      if(thetas.isEmpty) {}
      else{
        delta +=: deltas
        val del = (thetas.last.t * delta) :* as.last :* (as.last.map(1.0-_)) 
        //val del = (thetas.last.t * delta)
        getDelta(del, thetas.init, as.init)
      }
    }
    getDelta(y_hat-y,this.get_thetas,this.as.toList.init)
  }
  
  def get_gradient(x:Array[DenseVector[Double]], y:Array[DenseVector[Double]],lambda_reg:Double) :Unit ={
    reset_gradient
    if(x.size != y.size) {println("size of X does NOT match that of y");break}
    else {
        for(i<- 0 until x.size) {
          this.backProp(this.forwardProp(x(i),this.get_thetas), y(i))
          this.accumulate_Ds
        }
        
        for(i <- 0 until ds.size) {
          ds(i) = (ds(i) + this.thetas(i).map(_*lambda_reg)).map(_/x.size.toDouble)
        }
    }
  }
  
  def update_thetas(lambda:Double) :Unit ={
    for(i <- 0 until this.thetas.size) {
      thetas(i) -= ds(i).map(_*lambda)
    }
  }
  
  def optimize_gradient(x:Array[DenseVector[Double]], y:Array[DenseVector[Double]],lambda:Double,lambda_reg:Double,iter:Int) : List[DenseMatrix[Double]] ={
    for(i <- 1 to iter) {
      println(s"======================iteration $i===========================")
      get_gradient(x,y,lambda_reg)
      update_thetas(lambda)
      //this.print_thetas
      println(s"cost is ${get_cost(x,y,this.get_thetas)}")
    }
    this.get_thetas
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
      for(i <- 0 until X.size) {
        cost += (this.forwardProp(X(i),thetas)-y(i)).sumOfSquares
      }
      cost += this.get_thetas_unrolled.map(Math.pow(_,2)).sum * lambda
      cost/2/X.size
    }
  }
  
  def gradient_checking(X:Array[DenseVector[Double]], y:Array[DenseVector[Double]],epsilong:Double,lambda:Double) : Unit={
    val gradient = ArrayBuffer.empty[Double]
    val thetas_unrolled = this.unrolling_thetas(this.get_thetas)
    this.get_gradient(X, y,lambda)
    
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

