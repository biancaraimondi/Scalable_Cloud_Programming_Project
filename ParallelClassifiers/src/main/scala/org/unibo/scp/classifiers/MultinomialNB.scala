package org.unibo.scp.classifiers

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

class MultinomialNB(spark: SparkSession) extends Serializable{
  var probMap : collection.Map[String, Int] = Map.empty[String,Int]
  var colName : Array[String] = Array[String]()
  var allLabels : Array[String] = Array[String]()
  var labelColumn : Int = 0

  def train(rdd: RDD[Array[String]]): Unit ={
    colName = rdd.first()
    val trainSet = rdd.filter(line => !line.sameElements(colName))
    labelColumn = rdd.first().length
    allLabels = trainSet.map(row => row(labelColumn)).distinct().collect()

    /**MAP:
     *in each partition of the training set we need to analyse each row and add a tuple (key, 1)
     * the key will be label or a string composed by variable name, variable value and label*/
    //zero frequency problem not handled
    //continuous data (gaussian?) not handled
    val mapped = trainSet.flatMap(row =>{
      var seq = Seq.empty[(String, Int)]
      seq = seq :+ (""+row(labelColumn), 1)
      for (i<-0 until row.length) {
        if(i != labelColumn){
          val attibuteName = colName(i)
          val attributeValue = row(i)
          val key = attibuteName + "-" + attributeValue + "-" + row(labelColumn)
          seq = seq :+ (key, 1)
        }
      }
      seq
    })

    /**REDUCE
     * simply reduce by key summing over the value with the same key*/
    probMap = mapped.reduceByKey(_+_).collectAsMap()

  }

  def predict(row: Array[String]) ={
    val probVector = allLabels.map(label => {
      val currentLabel = label
      val freqCurrentLabel:Int = probMap.getOrElse(currentLabel,0)
      var probRowLabel:Float = freqCurrentLabel
      for (i <- 0 until row.length){
        if (i != labelColumn){
          val attributeName = colName(i)
          val attributeValue = row(i)
          val key = attributeName+"-"+attributeValue+"-"+currentLabel
          val freqEvent:Int = probMap.getOrElse(key,0)
          val probEvent:Float = freqEvent.toFloat/freqCurrentLabel
          probRowLabel *= probEvent
        }
      }
      (label, probRowLabel)
    })
    val maxProb = probVector.maxBy(_._2)._1
    maxProb
  }



  def score(rdd: RDD[Array[String]]) ={
    val testSet = rdd.filter(line => !line.sameElements(colName))

    testSet.map(row => {
      val maxProb = predict(row)
      val correctLabel = ""+row(labelColumn)
      (correctLabel+","+maxProb+","+correctLabel.equals(maxProb),1)
    })
  }
}
