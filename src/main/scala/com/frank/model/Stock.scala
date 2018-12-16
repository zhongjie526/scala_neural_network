package com.frank.model
import scala.util.Try
import breeze.linalg.DenseVector

case class Stock(symbol:String,price:Double,market_cap:Double,data:Data){

  def this(t: (String, Double,Double, Data)) = {
    this(t._1, t._2, t._3, t._4)
  }
  
  def this(line:String) = {
    this(Stock.init(line))
  }
    
}

object Stock{
  private def init(line:String)={
    val fields = line.split(",",-1)
    val symbol = fields(0)
    val numericals = fields.tail.map{field:String=>if(field.length()>0 && field.head=='C') field.tail else field}
    .map{field:String=>Try(field.toDouble).toOption.getOrElse(Double.NaN)}
    
    val price = numericals(0)
    val market_cap = numericals(1)
    val data = new Data(numericals.drop(2))
    
    (symbol,price,market_cap,data)
  }
}
  
