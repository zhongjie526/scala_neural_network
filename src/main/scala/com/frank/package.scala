
package com
import java.sql.Timestamp
import java.util.Date
import breeze.linalg._
import breeze.stats.mean
import scala.util.control.Breaks._

package object frank {
  
  def get_timestamp:Long = new Date().getTime
	
	def to_ts(timestamp:Long) = new Timestamp(timestamp).toString
	
	def get_ts = to_ts(get_timestamp)
	
	val unit = (x:Double)=>x
	
	val sigmoid = (x:Double)=>1.0/(1.0+Math.exp(-x))
	
	val relu = (x:Double) => Math.max(0.0, x)
	
	val softplus = (x:Double) => Math.log(1.0+Math.exp(x))
	
  implicit class stats[T](series:scala.collection.Traversable[T])(implicit number:Numeric[T]) {
    val s = series.map(number.toDouble)
    val n = s.size
    val sample_mean = s.sum/n
    
    val sample_median = {
      val s_sorted= s.toSeq.sorted
      if(n%2==1) s_sorted(n/2)
      else (s_sorted(n/2-1)+s_sorted(n/2))/2
    }
    
    def getSecondCentralM = s.map(x=>scala.math.pow(x-sample_mean,2)).sum/n
    
    def getStd = scala.math.sqrt(getSecondCentralM*n/(n-1))
    
    def getThirdCentralM = s.map(x=>scala.math.pow(x-sample_mean,3)).sum/n
    
    def getFourthCentralM = s.map(x=>scala.math.pow(x-sample_mean,4)).sum/n
    
    def getSkewness = getThirdCentralM*n*n/(n-1)/(n-2)/scala.math.pow(getStd,3)
    
    def getKurtosis = getFourthCentralM/scala.math.pow(getStd,4)*n*n*(n+1)/(n-1)/(n-2)/(n-3)
    
    def getExcessKurtosis = getKurtosis-3.0*(n-1)*(n-1)/(n-2)/(n-3)
    
    def getSarleBimodality = (scala.math.pow(getSkewness,2)+1)/getKurtosis
    
    def getUniMode = math.sqrt(5.0/3)*math.abs(sample_median-sample_mean)/getStd
  }
  
  implicit class ListOps(v:DenseVector[Double]) {

    def sumOfSquares = v.map{case x=>x*x}.sum
    
    def sumOfAbs = v.map(Math.abs(_)).sum
    
    def sst = v.map{x=>x-mean(v)}.sumOfSquares
    
    def avg :Double={
      val w = v.toArray.filterNot(_.isNaN)
      if(w.isEmpty) Double.NaN
      else breeze.stats.mean(DenseVector(w))
    }
    
    def std :Double={
      val w = v.toArray.filterNot(_.isNaN)
      if(w.isEmpty) Double.NaN
      else breeze.stats.stddev(DenseVector(w))
    }
    

  }
        def time[T](block: => T): T = {
    val start = System.currentTimeMillis
    val res = block
    val totalTime = System.currentTimeMillis - start
    println("Elapsed time: %.2f s".format(totalTime/1000.0))
    res
}
  

  

}

