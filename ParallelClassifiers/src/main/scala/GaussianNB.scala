import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

class GaussianNB(spark : SparkSession) extends Serializable{

  var reducedAttribute : collection.Map[String, (Double, Double)] = Map.empty
  var reducedLabel : collection.Map[String, Int] = Map.empty[String,Int]
  var labelColumn : Int = 0
  var colName : Array[String] = Array[String]()
  var allLabels : Array[String] = Array[String]()

  def train(rdd: RDD[Array[String]]): Unit ={

    val trainSet = rdd.filter(line => !line.sameElements(colName))
    labelColumn = rdd.first().length - 1
    allLabels = trainSet.map(row => row(labelColumn)).distinct().collect()

    val mappedLabel = trainSet.map(row => {
      val labelRow = "" + row(labelColumn)
      (labelRow,1)
    })
    reducedLabel = mappedLabel.reduceByKey(_+_).collectAsMap()

    //mean and std deviation
    val mappedAttribute = trainSet.flatMap(row => {
      var seq = Seq.empty[(String, Double)]
      val label = row(labelColumn)
      for (i <- 0 until row.length){
        if(i != labelColumn){
          val attributeName = colName(i)
          val attributeValue: Double = row(i).toDouble
          val key = attributeName + "-" + label

          seq = seq :+ (key, attributeValue)
        }
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

  def rowLabelProb(row: Array[String], label : String): (String, Double) ={

    val freqCurrentLabel:Int = reducedLabel.getOrElse(label, 0)
    var probRowLabel:Double = freqCurrentLabel
    for(i<-0 until row.length){
      if(i != labelColumn){
        val key = colName(i)+"-"+label
        val mean = reducedAttribute.getOrElse(key, (0.0,0.0))._1
        val stdev = reducedAttribute.getOrElse(key, (0.0, 0.0))._2
        val x = row(i).toDouble
        val p = (1.0/(stdev * scala.math.sqrt(2*scala.math.Pi))
          * scala.math.exp(-0.5*scala.math.pow((x-mean)/stdev,2)))
        probRowLabel *= p
      }
    }

    return (label, probRowLabel)
  }

  def predict(row: Array[String]): String ={
    val probVector = allLabels.map(label => {
      rowLabelProb(row, label)
    })

    val maxProb = probVector.maxBy(_._2)._1
    return maxProb
  }

  def score(rdd:RDD[Array[String]]): RDD[(String, Int)] ={
    val testSet = rdd.filter(line => !line.sameElements(colName) )

    val prediction = testSet.map(row => {
      val maxProb: String = predict(row)
      val correctLabel = ""+row(labelColumn)
      (correctLabel+","+maxProb+","+correctLabel.equals(maxProb),1)
    })
    val score = prediction.reduceByKey(_+_)

    return score
  }


}