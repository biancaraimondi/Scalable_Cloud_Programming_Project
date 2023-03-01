package classifiers

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

class GaussianNB(spark : SparkSession) extends Serializable{

  var colName : Array[String] = Array[String]()
  var allLabels : Array[Double] = Array[Double]()
  var reducedLabel : collection.Map[Double, Int] = Map.empty[Double,Int]

  var reducedAttribute : collection.Map[String, (Double, Double)] = Map.empty

  def train(trainSet: RDD[LabeledPoint]): Unit ={
    //train set don't contain the header row
    allLabels = trainSet.map(_.label).distinct().collect()

    val mappedLabel = trainSet.map(point => (point.label, 1))
    reducedLabel = mappedLabel.reduceByKey(_+_).collectAsMap()

    //mean and std deviation
    val mappedAttribute = trainSet.flatMap(point => {
      val features = point.features.toArray
      var seq = Seq.empty[(String, Double)]
      for (i <- 0 until features.length){
        val attributeName = colName(i)
        val attributeValue: Double = features(i).toDouble
        val key = attributeName + "-" + point.label
        seq = seq :+ (key, attributeValue)
      }
      seq
    })

    reducedAttribute = mappedAttribute.groupByKey().mapValues{ values =>
      val count = values.size
      val sum = values.sum
      val mean = sum/count.toDouble
      val stdev = new StandardDeviation().evaluate(values.toArray)
      (mean, stdev)
    }.collectAsMap()
  }

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

  def predict(row: Array[Double]): Double ={
    val probVector = allLabels.map(label => {
      rowLabelProb(row, label)
    })

    val maxProb = probVector.maxBy(_._2)._1
    return maxProb
  }

  def score(testSet:RDD[LabeledPoint]): RDD[(String, Int)] ={

    val prediction = testSet.map(point => {
      val maxProb: Double = predict(point.features.toArray)
      val correctLabel = point.label
      (correctLabel+","+maxProb+","+(maxProb == correctLabel),1)
    })
    val score = prediction.reduceByKey(_+_)
    return score
  }

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