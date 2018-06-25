package com.frank

import algorithms._
import algorithms.Utils._
import breeze.linalg._
import java.io._
import scala.util.Try
import java.util.Properties

object GradientChecking extends App {
  
  
  val prop = new Properties()
  val propIn = new FileInputStream("ann.properties")
  prop.load(propIn)
  propIn.close
  
  val working_dir = System.getProperty("user.dir")
  val arch:List[Int] = prop.getProperty("architecture").split("\\|").toList.map(_.toInt)
  
  
  val file = scala.io.Source.fromFile(s"$working_dir/${prop.getProperty("input.file.name")}").getLines().toList
  val input = file.map(line=>line.split(",",-1).tail
                                 .map{field:String=>if(field.length()>0 && field.head=='C') field.tail else field}
                                 .map{field:String=>Try(field.toDouble).toOption.getOrElse(Double.NaN)})
  

  
  val nn = ANN_Biased_Perc(arch,relu)
  val thetas_source = nn.initialize
 
  val (x_train,y_train) = input.toArray.map{xs => (DenseVector(xs.tail),DenseVector(xs.head))}.unzip

  val (x_train_scaled,means,stds) = scaling(x_train)
    
  
  nn.gradient_checking(x_train_scaled, y_train,thetas_source, 0.0001,0.000)
  
  
}