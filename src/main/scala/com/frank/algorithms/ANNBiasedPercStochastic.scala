package com.frank.algorithms
import com.frank._
import scala.util.control.Breaks._
import breeze.linalg._
import scala.collection.mutable.ArrayBuffer
import util.Random.shuffle

case class ANNBiasedPercStochastic(architecture: List[Int], actFunc: Double => Double) {

  override def toString = s"Neural Network with arhitecture ${architecture.map(_.toString).reduce(_ + " ==> " + _)}"

  private val rand = new scala.util.Random
  private val n_levels = architecture.size
  private val n_units = scala.collection.mutable.ArrayBuffer.empty[Int]

  var printCost = true

  def initialize: List[DenseMatrix[Double]] = {
    val thetas = scala.collection.mutable.ArrayBuffer.empty[DenseMatrix[Double]]
    for (l <- 0 until n_levels - 1) {
      thetas += DenseMatrix.rand(architecture(l + 1), architecture(l) + 1).map(_ * 2 - 1)
    }
    thetas.toList
  }

  def unrolling_thetas(thetas: List[DenseMatrix[Double]]): Array[Double] = {
    val thetas_unrolled = ArrayBuffer.empty[Double]
    thetas.foreach(T => thetas_unrolled ++= T.toArray)
    thetas_unrolled.toArray
  }

  def rolling_thetas(thetas_unrolled: Array[Double]): List[DenseMatrix[Double]] = {
    val thetas = ArrayBuffer.empty[DenseMatrix[Double]]

    def roll(n_level_rolled: Int, thetas_unrolled: Array[Double]): Unit = {
      if (n_level_rolled >= n_levels) {}
      else {
        val n_rows = architecture(n_level_rolled)
        val n_cols = architecture(n_level_rolled - 1) + 1
        val n_rolling = n_rows * n_cols
        thetas += DenseMatrix(thetas_unrolled.take(n_rolling)).reshape(n_rows, n_cols)
        roll(n_level_rolled + 1, thetas_unrolled.drop(n_rolling))
      }
    }

    roll(1, thetas_unrolled)
    thetas.toList
  }

  def forwardProp(x: DenseVector[Double], thetas: List[DenseMatrix[Double]]): (DenseVector[Double], List[DenseVector[Double]], List[DenseVector[Double]]) = {

    if (x.size != this.architecture.head) {
      println("size of input does Not fit the architecture of the neural network")
      println(s"${this.architecture.head} input nodes expected but ${x.size} observed")
      break
    }

    def getOutput(spikes: List[DenseVector[Double]], thetas: List[DenseMatrix[Double]], acts: List[DenseVector[Double]]): (DenseVector[Double], List[DenseVector[Double]], List[DenseVector[Double]]) = {
      val activation = DenseVector(1.0 +: spikes.last.map(actFunc).toArray)

      //if(thetas.isEmpty) (spikes.last,spikes,acts )
      if (thetas.isEmpty) (spikes.last.map(actFunc), spikes, acts) //applying activation function to the output here
      else {
        val spike = thetas.head * activation
        getOutput(spikes :+ spike, thetas.tail, acts :+ activation)
      }
    }
    getOutput(List(x), thetas.toList, List.empty[DenseVector[Double]])
  }

  def forwardProp(X: Array[DenseVector[Double]], thetas: List[DenseMatrix[Double]]): Array[DenseVector[Double]] = {
    X.map { x =>
      val (pred, _, _) = forwardProp(x, thetas)
      pred
    }
  }

  def backProp(y_hat: DenseVector[Double], y: DenseVector[Double], thetas: List[DenseMatrix[Double]], acts: List[DenseVector[Double]]): List[DenseVector[Double]] = {
    //deltas.clear

    if (y.size != this.architecture.last) { println("size of output does Not fit the architecture of the neural network"); break }
    if (y.size != y_hat.size) { println("size of output does NOT match the size of the predicted output"); break }

    def getDelta(deltas: List[DenseVector[Double]], thetas: List[DenseMatrix[Double]], as: List[DenseVector[Double]]): List[DenseVector[Double]] = {

      if (thetas.isEmpty) deltas.tail
      else {
        //val del = DenseVector(((thetas.last.t * deltas.head) :* as.last :* (as.last.map(1.0-_))).toArray.tail)
        val del = DenseVector(((thetas.last.t * deltas.head) :* (as.last.map { a => if (a > 0) 1.0 else 0.0 })).toArray.tail)
        //val del = (thetas.last.t * deltas.head) :* as.last.map(sigmoid(_))
        getDelta(del +: deltas, thetas.init, as.init)
      }
    }
    getDelta(List(((y_hat - y).map(Math.signum(_)) / y)), thetas, acts)
  }

  def get_gradient(thetas: List[DenseMatrix[Double]], x: Array[DenseVector[Double]], y: Array[DenseVector[Double]], lambda_reg: Double): List[DenseMatrix[Double]] = {

    if (x.size != y.size) { println("size of X does NOT match that of y"); break }
    else {

      val big_ds = x.zip(y).par.map {
        case (a, b) =>
          val (temp_pred, temp_spikes, temp_acts) = this.forwardProp(a, thetas)
          val temp_deltas = this.backProp(temp_pred, b, thetas, temp_acts)
          temp_deltas.zip(temp_acts).map { case (d, act) => d * act.t }
      }.reduce { (x, y) => x.zip(y).map { case (a, b) => a + b } }

      big_ds.zip(thetas).par.map { case (d, t) => (d + (t :* lambda_reg)) :/ x.size.toDouble }.toList

    }
  }

  def update_thetas(big_ds: List[DenseMatrix[Double]], thetas: List[DenseMatrix[Double]], lambda: Double): List[DenseMatrix[Double]] = {
    thetas.zip(big_ds).par.map { case (t, d) => t - d.map(_ * lambda) }.toList
  }

  def optimize_gradient(thetas: List[DenseMatrix[Double]], x: Array[DenseVector[Double]], y: Array[DenseVector[Double]], lambda: Double, lambda_reg: Double, n_iter: Int, n_batch: Int): List[DenseMatrix[Double]] = {
    def update_gradient(thetas: List[DenseMatrix[Double]], x: Array[DenseVector[Double]], y: Array[DenseVector[Double]], lambda: Double, lambda_reg: Double, iter: Int, batch_no: Int): List[DenseMatrix[Double]] = {

      if (iter <= n_iter) {

        val x_batch = x.slice(n_batch * batch_no, n_batch * (batch_no + 1))
        val y_batch = y.slice(n_batch * batch_no, n_batch * (batch_no + 1))

        if (x_batch.isEmpty) {
          val (x_new, y_new) = shuffle(x.zip(y).toSeq).toArray.unzip
          update_gradient(thetas, x_new, y_new, lambda, lambda_reg, iter + 1, 0)
        } else {

          val big_ds = get_gradient(thetas, x_batch, y_batch, lambda_reg)

          // if(batch_no<1) println(x_batch.map(_.toString).reduce(_+"|"+_))

          if (batch_no <= x.length / n_batch) {
            if (printCost) {
              if (iter % 100 == 0) {
                println(s"======================iteration $iter batch $batch_no with batch size = ${x_batch.size}===========================")
                println(s"cost is ${get_cost_regularized(x_batch, y_batch, thetas, lambda_reg)}")
              }
            }
            update_gradient(update_thetas(big_ds, thetas, lambda), x, y, lambda, lambda_reg, iter, batch_no + 1)
          } else {
            val (x_new, y_new) = shuffle(x.zip(y).toSeq).toArray.unzip
            update_gradient(update_thetas(big_ds, thetas, lambda), x_new, y_new, lambda, lambda_reg, iter + 1, 0)
          }
        }
      } else thetas

    }
    update_gradient(thetas, x, y, lambda, lambda_reg, 1, 0)

  }

  def get_rsquare(x: Array[DenseVector[Double]], y: Array[DenseVector[Double]], thetas: List[DenseMatrix[Double]]): DenseVector[Double] = {
    val y_h = this.forwardProp(x, thetas)
    val sse = y_h.zip(y).map { case (y_h, y) => (y_h - y) }.reduce { (a, b) => a + (b :^ 2.0) }

    val y_mean = y.reduce(_ + _) / y.size.toDouble
    val sst = y.map(v => (v - y_mean)).reduce { (a, b) => a + (b :^ 2.0) }
    (sst - sse) / sst
  }

  def get_cost_regularized(X: Array[DenseVector[Double]], y: Array[DenseVector[Double]], thetas: List[DenseMatrix[Double]], lambda_reg: Double): Double = {
    if (X.size != y.size) { println("size of X does NOT match that of y"); break }
    else {
      var cost: Double = 0

      cost += X.zip(y).par.map {
        case (x, y) =>
          val (pred, _, _) = this.forwardProp(x, thetas)
          ((pred - y) / y).sumOfAbs
      }.sum
      cost += this.unrolling_thetas(thetas).map(Math.pow(_, 2)).sum * lambda_reg
      cost / X.size
    }
  }

  def gradient_checking(X: Array[DenseVector[Double]], y: Array[DenseVector[Double]], thetas: List[DenseMatrix[Double]], epsilong: Double, lambda: Double): Unit = {
    val gradient = ArrayBuffer.empty[Double]
    val thetas_unrolled = this.unrolling_thetas(thetas)
    val big_ds = this.get_gradient(thetas, X, y, lambda)

    for (i <- 0 until thetas_unrolled.size) {
      val thetas_unrolled_plus = thetas_unrolled.clone
      val thetas_unrolled_minus = thetas_unrolled.clone
      thetas_unrolled_plus(i) += epsilong
      thetas_unrolled_minus(i) -= epsilong
      val thetas_plus = this.rolling_thetas(thetas_unrolled_plus)
      val thetas_minus = this.rolling_thetas(thetas_unrolled_minus)
      val cost_plus = this.get_cost_regularized(X, y, thetas_plus, lambda)
      val cost_minus = this.get_cost_regularized(X, y, thetas_minus, lambda)
      gradient += (cost_plus - cost_minus) / 2 / epsilong
    }

    (gradient.toArray zip this.unrolling_thetas(big_ds)).foreach(println)

  }

}

