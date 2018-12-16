package com.frank

import java.io.FileInputStream
import java.util.Properties
import com.frank.model.Stock
import java.nio.file.Paths
import java.nio.file.Files
import java.io.Reader
import com.opencsv.CSVReader;

object Simulate extends App {
    val prop = new Properties()
    val propIn = new FileInputStream("ann.properties")
    prop.load(propIn)
    propIn.close
    val working_dir = System.getProperty("user.dir")
  
//    val file = scala.io.Source.fromFile(s"$working_dir/${prop.getProperty("input.file.name")}").getLines().toList
//    
//    println(file.count(_=> true))
//    
//    file.foreach(line => {
//      val stock = new Stock(line)
//      println(stock.symbol + " " +stock.price + " " +stock.market_cap +" " +stock.data)
//      }
//    )
    
    val reader: Reader = Files.newBufferedReader(Paths.get(s"$working_dir/${prop.getProperty("input.file.name")}"));
    
    val csvReader: CSVReader = new CSVReader(reader)
    
    var record:Array[String] = Array[String]()
    
    while({record =csvReader.readNext();record!=null}){
      println(record(0))
    }
}