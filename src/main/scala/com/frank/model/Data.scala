package com.frank.model

import breeze.linalg.DenseVector

case class Data(values:DenseVector[Double]) {
  
  def this(v:Array[Double])={
    this(DenseVector(v))
  }
  

  
}