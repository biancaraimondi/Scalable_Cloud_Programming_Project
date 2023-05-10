package org.unibo.scp.classifiers

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
* This class implements the Gaussian Naive Bayes algorithm
* @author Gabriele Pasquali
*/
class GaussianNB(spark : SparkSession) extends Serializable{
  /** List of the attribute name as Array of String */  
  var colName : Array[String] = Array[String]()
  /** List of all the Label available in the Dataset as Double */
  var allLabels : Array[Double] = Array[Double]()
  /** Map used to collect summarized info of the label.
  * The key is the label name and then value is the absolute frequency */
  var reducedLabel : collection.Map[Double, Int] = Map.empty[Double,Int]
  /** Map used to collect summarized info of attribute in form of [string, (Double, Double)].
  * The key is a String composed by concatenation of attribute name (obtained from colName) and the label.
  * The value has the mean in the first position and stdDev in the second */
  var reducedAttribute : collection.Map[String, (Double, Double)] = Map.empty

  /** 
  * This method is used to train the model, during this phase we initialize all the class field.
  * @param trainSet as RDD of LabeledPoint
  * @return Nothing 
  */
  def train(trainSet: RDD[LabeledPoint]): Unit ={
    //trainSet doesn't contain the header row
    allLabels = trainSet.map(_.label).distinct().collect()
    //with the LabeledPoint we can easily obtain the label and map it
    val mappedLabel = trainSet.map(point => (point.label, 1))
    //reducing by key (label) and sumUp to obtain absolute frequency
    reducedLabel = mappedLabel.reduceByKey(_+_).collectAsMap()

    val mappedAttribute = trainSet.flatMap(point => {
      val features = point.features.toArray
      var seq = Seq.empty[(String, Double)]
      //for each attribute of the point
      for (i <- 0 until features.length){
        val attributeName = colName(i)
        val attributeValue: Double = features(i).toDouble
        val key = attributeName + "-" + point.label
        //add to seq (attributeName-label, attributeValue) 
        seq = seq :+ (key, attributeValue)
      }
      seq
    })

    //mean and std deviation
    reducedAttribute = mappedAttribute.groupByKey().mapValues{ values =>
      val count = values.size
      val sum = values.sum
      val mean = sum/count.toDouble
      val stdev = new StandardDeviation().evaluate(values.toArray)
      (mean, stdev)
    }.collectAsMap()
  }

  /**
  * This method is used to calculate the Gaussian probability of a row to belong to specific label  
  * using reducedLabel and reducedAttribute  
  * @param row
  * @param label
  * @return (label, probRowLabel) 
  */
  def rowLabelProb(row: Array[Double], label : Double): (Double, Double) ={
    val freqCurrentLabel:Int = reducedLabel.getOrElse(label, 0)
    var probRowLabel:Double = freqCurrentLabel
    for(i<-0 until row.length){
      val key = colName(i)+"-"+label
      val mean = reducedAttribute.getOrElse(key, (0.0,0.0))._1
      val stdev = reducedAttribute.getOrElse(key, (0.0, 0.0))._2
      val x = row(i).toDouble
      val p = (1.0/(stdev * scala.math.sqrt(2*scala.math.Pi))
        * scala.math.exp(-0.5*scala.math.pow((x-mean)/stdev,2)))
      probRowLabel *= p
    }

    return (label, probRowLabel)
  }
  
  /** 
  * This method is used to calculate which is the most prob label of a row.
  * It uses rowLabelProb to calc the single label prob and select the highest. 
  * @param row
  * @return maxProb is the most prob label */
  def predict(row: Array[Double]): Double ={
    //create a map of (label, prob)
    val probVector = allLabels.map(label => {
      rowLabelProb(row, label)
    })

    //select the label of the couple with max probability
    val maxProb = probVector.maxBy(_._2)._1
    return maxProb
  }

  /**
  * This method calculate the prediction of each row of input testSet, evaluate it and return the score
  in form of RDD String-Int. 
  * @param testSet
  * @return score
   */
  def score(testSet:RDD[LabeledPoint]): RDD[(String, Int)] ={

    val prediction = testSet.map(point => {
      val maxProb: Double = predict(point.features.toArray)
      val correctLabel = point.label
      (correctLabel+","+maxProb+","+(maxProb == correctLabel),1)
    })
    //we obtain something similar to a confusion matrix
    val score = prediction.reduceByKey(_+_)
    return score
  }

  //same procedure of score but we obtain only the accuracy score
  def accuracy(testSet:RDD[LabeledPoint]): Double ={

    val prediction = testSet.map(point => {
      val maxProb: Double = predict(point.features.toArray)
      val correctLabel = point.label
      (maxProb == correctLabel,1)
    })

    val score = prediction.reduceByKey(_+_).collectAsMap()
    val correct = score.getOrElse(true, 0).toDouble
    val wrong = score.getOrElse(false, 0)
    val tot = correct+wrong

    return correct/tot
  }

}
