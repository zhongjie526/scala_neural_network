package com.frank.algorithms

import com.frank._
import breeze.linalg._
import breeze.stats._
import scala.util.control.Breaks._

object Utils {
  
  val rand = new scala.util.Random
  
  def reluFunc(x:Double) :Double = if(x>0) x else 0
  
  def reluFunc(x:DenseVector[Double]) : DenseVector[Double] =  x.map(reluFunc)
  
  def sigmoidFunc(x:Double) :Double = 1/(1+math.exp(-x))
  
  def sigmoidFunc(x:DenseVector[Double]) :DenseVector[Double] = x.map(sigmoidFunc)
  
  def costFunc(y:DenseVector[Double], X:DenseMatrix[Double], theta:DenseVector[Double]):Double ={
    (X * theta -y).sumOfSquares/y.size/2
  }

  
  def scaling(x:Array[DenseVector[Double]]) ={
    val n_rows = x.size
    val n_cols = x(0).size
    val data_br = DenseMatrix(x.map(_.toArray).reduce(_++_)).reshape(n_cols,n_rows).t(::,*)
    val means = (data_br.map(_.avg)).t
    val stds = (data_br.map(_.std)).t
    val x_scaled = x.map{v=>(v-means):/stds}
    (x_scaled,means,stds)
  }
  
  def scaling(v:DenseVector[Double],means:DenseVector[Double],stds:DenseVector[Double]) : DenseVector[Double]={
      (v-means):/stds
     // w.map(y=>if(y.isNaN) 0.0 else y
     
  }
  
  def scaling(x:Array[DenseVector[Double]],means:DenseVector[Double],stds:DenseVector[Double]) : Array[DenseVector[Double]]={
    x.map{v=>scaling(v,means,stds)}
      
  }
  
  def randomize(x:DenseVector[Double]):DenseVector[Double]={
    x.map{v=>
      val r = scala.util.Random
      val s = scala.util.Random
      if(v.isNaN) r.nextGaussian()/3.0 else v*(1.0+s.nextGaussian()/50.0)
    }
  }
  
  def randomize(x:Array[DenseVector[Double]]):Array[DenseVector[Double]]={
    x.map(randomize(_))
  }
  
  def simulate(x:DenseVector[Double],n:Int): Array[DenseVector[Double]] = {
    Array.fill(n)(randomize(x))
  }

  def costFuncFirstDerivative(y:DenseVector[Double], X:DenseMatrix[Double], theta:DenseVector[Double]): DenseVector[Double] ={
    val W = X(::,*) :* (X * theta - y)
    mean(W(::,*)).t
  }
  
  def R2(input:DenseMatrix[Double], theta:DenseVector[Double], scaling:DenseVector[Double]): Double ={
      val y = input(::,0)
      val x = input(::,1 to -1)
      val x_scaled = x(*,::) :/ scaling
      val x_intercept =  DenseMatrix.horzcat(DenseMatrix.fill(x_scaled.rows,1)(1.0),x_scaled)
      val y_hat = x_intercept * theta
      val sse = (y-y_hat).map(Math.pow(_,2)).sum
      val sst = y.sst
      1-sse/sst
  }

  def splitTrainTest[T](input:List[T],training : Double) : (List[T],List[T]) ={
    val n = input.size
    val n_train :Int = (n*training).toInt
    val output = scala.util.Random.shuffle(input)
    (output.slice(0,n_train),output.slice(n_train,n))
  }
  
 

  
  def gradientDescent(Data:DenseMatrix[Double],iter:Int,alpha:Double,batch:Int) :(DenseVector[Double],DenseVector[Double]) = {
    var theta:DenseVector[Double] = DenseVector.fill(Data.cols)(rand.nextDouble*2-1)
    val stds:DenseVector[Double] = stddev(Data(::,*)).t

    var cost:Double = 0.0
    
    val lst_cost = scala.collection.mutable.Queue.empty[Double]
    
    breakable{
      for(i <-1 to iter){
        for(j <- 0 until Data.rows-1 by batch) {
          val Data_batch = Data(j until j+batch,::)
          val y = Data_batch(::,0)
          val X_n = Data_batch(::,1 to -1)
          val X_scaled = X_n(*,::) :/ stds(1 to -1)
          val X = DenseMatrix.horzcat(DenseMatrix.fill(Data_batch.rows,1)(1.0),X_scaled)
          if(X.cols != theta.size) {println("size of x is incorrect"); break}
          if(lst_cost.size < 10000/batch) lst_cost += costFunc(y,X,theta)
          else {lst_cost.dequeue; lst_cost += costFunc(y,X,theta)}
          val cost_avg = lst_cost.sample_mean
          val alpha_modified:Double = alpha*1000/(1000+iter)
          println(s"iteration = $i, batch = $j : cost = $cost_avg, alpha = $alpha_modified, theta = $theta")
          val adjustment:DenseVector[Double]= costFuncFirstDerivative(y,X,theta)
          if(theta.size != adjustment.size) {println(s"size of adjustment is ${adjustment.size}"); break}
          theta = theta - adjustment.map(_*alpha_modified)
  
        }
      }
    }
    (stds,theta)
  }
}