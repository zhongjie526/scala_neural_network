package com.frank

import algorithms._
import algorithms.Utils._
import breeze.linalg._
import java.io._
import scala.util.Try
import java.util.Properties

object NormApp extends App {
  
  val prop = new Properties()
  val propIn = new FileInputStream("ann.properties")
  prop.load(propIn)
  propIn.close
  
  val iterations:Int = prop.getProperty("iterations").toInt
  val lambda:Double = prop.getProperty("lambda").toDouble 
  val lambda_reg:Double = prop.getProperty("lambda.regularization").toDouble 
  val initialize:Boolean = prop.getProperty("initialize").toBoolean
  val initialize_scaling:Boolean = prop.getProperty("initialize_scaling").toBoolean
  val arch:List[Int] = prop.getProperty("architecture").split("\\|").toList.map(_.toInt)
  val randomization:Double = prop.getProperty("randomization").toDouble 
  
  val working_dir = System.getProperty("user.dir")
  
  val file = scala.io.Source.fromFile(s"$working_dir/${prop.getProperty("input.file.name")}").getLines().toList
  val input = file.map(line=>line.split(",",-1).tail
                                 .map{field:String=>if(field.length()>0 && field.head=='C') field.tail else field}
                                 .map{field:String=>Try(field.toDouble).toOption.getOrElse(Double.NaN)})

  val nn = ANN_Biased_Norm(arch,sigmoid)

  lazy val thetas_source = scala.io.Source.fromFile(s"$working_dir/thetas_output").getLines().toArray.map(_.toDouble)
  val thetas_initial = if(initialize) nn.initialize else  nn.rolling_thetas(thetas_source)
  
  nn.printCost = prop.getProperty("cost.print").toBoolean
  
 
  val (x_train,y_train) = input.toArray.map{xs => (DenseVector(xs.tail),DenseVector(xs.head))}.unzip
  
  
  if(initialize_scaling) {
    val (x_train_scaled,means_new,stds_new) = scaling(x_train)
    val means_file = new File(s"$working_dir/means")
    val bw_means = new BufferedWriter(new FileWriter(means_file))
    means_new.foreach(x=>bw_means.write(x+"\n"))
    bw_means.close
  
    val stds_file = new File(s"$working_dir/stds")
    val bw_stds = new BufferedWriter(new FileWriter(stds_file))
    stds_new.foreach(x=>bw_stds.write(x+"\n"))
    bw_stds.close
  }
  

  val means = DenseVector(scala.io.Source.fromFile(s"$working_dir/means").getLines().map(_.toDouble).toArray)
  val stds = DenseVector(scala.io.Source.fromFile(s"$working_dir/stds").getLines().map(_.toDouble).toArray)
  val x_train_scaled = scaling(x_train,means,stds)
  
  
  val thetas =  nn.optimize_gradient(thetas_initial,x_train_scaled, y_train, lambda ,lambda_reg, iterations)
  
  val thetas_unrolled = nn.unrolling_thetas(thetas)
 


  val estimates = file.toArray.map{line=>
    val id = line.split(",",-1).head
    val data = line.split(",",-1).tail.map{field:String=>if(field.length()>0 && field.head=='C') field.tail else field}
    .map{field:String=>Try(field.toDouble).toOption.getOrElse(Double.NaN)}
    val y = data.head
    val x = scaling(DenseVector(data.tail),means,stds)
    val (pred_y,_,_)=nn.forwardProp(x, thetas)
    val pred = pred_y(0)
    val diff = pred/y-1

    (id,y,pred,diff)
  }
  
  val long = new File(s"$working_dir/result.csv")
  val bw_long = new BufferedWriter(new FileWriter(long))
  
  estimates.sortBy(-_._4).foreach{case (id,y,pred,diff) => 
  bw_long.write(id.toString+","+y.toString+","+pred.toString+","+diff.toString +"\n")}
  bw_long.close()
  

  val thetas_output = new File(s"$working_dir/thetas_output")
  val bw_thetas = new BufferedWriter(new FileWriter(thetas_output))

  thetas_unrolled.foreach(t=>bw_thetas.write(t.toString+"\n"))
  bw_thetas.close()
  
  val thetas_transposed = new  File(s"$working_dir/thetas_transposed")
  val bw_thetas_t = new BufferedWriter(new FileWriter(thetas_transposed))
  nn.unrolling_thetas(thetas.map(_.t)).foreach(t=>bw_thetas_t.write(t.toString+"\n"))
  bw_thetas_t.close()
  
  
  for(i<-1 to thetas.size){
    val theta_file = new File(s"$working_dir/theta_$i")
    val bw_theta_file = new BufferedWriter(new FileWriter(theta_file))
    val theta = thetas(i-1)
    theta(*,::).map(_.toVector).foreach(t=> bw_theta_file.write(t.map(_.toString).reduce(_+","+_)+"\n"))
    bw_theta_file.close()
  }

  val train_r2 = nn.get_rsquare(x_train_scaled,y_train,thetas)
  println(s"training r2 = $train_r2")


}
