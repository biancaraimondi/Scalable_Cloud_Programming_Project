import org.apache.spark.sql.SparkSession
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation


object Main {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .master("local[1]")
      .appName("SparkByExamples.com")
      .getOrCreate()

    //    val rddFromFile = sc.textFile("src/main/data/data.csv")
    val rddFromFile = spark.sparkContext.textFile("src/main/data/data.csv")
    val rdd = rddFromFile.map(f => {
      f.split(",")
    })
    val labelColumn = 7
    val colName = rdd.first()
    colName.foreach(println)
    val data = rdd.filter(line => !(line.sameElements(colName)))


    //get every possible label
    val allLabels = data.map(row => row(labelColumn)).distinct().collect()
    //    allLabels.foreach(println)


    //split into test and train set
    val Array(trainSet, testSet) = data.randomSplit(Array(0.7,0.3))

    val mappedLabel = trainSet.map(row => {
      val labelRow = "" + row(labelColumn)
      (labelRow,1)
    })


    val reducedLabel = mappedLabel.reduceByKey(_+_).collectAsMap()

    //mean and std deviation
    val mappedAttribute = data.flatMap(row => {
      var seq = Seq.empty[(String, Double)]
      val label = row(labelColumn)
      //si potrebbe usare row.map ma viene mappata anche la colonna della label
      //PROBLEMA: come ricavare il nome della colonna che stiamo legendo?
      for (i <- 3 until 7){
        if(i != labelColumn){
          val attributeName = colName(i)
          val attributeValue: Double = row(i).toDouble
          val key = attributeName + "-" + label

          seq = seq :+ (key, attributeValue)
        }
      }
      seq
    })

    val reducedAttribute = mappedAttribute.groupByKey().mapValues{ values =>
      val count = values.size
      val sum = values.sum
      val mean = sum/count.toDouble
      val stdev = new StandardDeviation().evaluate(values.toArray)
      (mean, stdev)
    }.collectAsMap()

    reducedAttribute.foreach(println)

    val predict = testSet.map(row => {
      val probVector = allLabels.map(label => {
        val currentLabel = label
        val freqCurrentLabel:Int = reducedLabel.getOrElse(label, 0)
        var probRowLabel:Double = freqCurrentLabel
        for (i<-3 until 7){
          if(i != labelColumn){
            val key = colName(i)+"-"+currentLabel
            val mean = reducedAttribute.getOrElse(key, (0.0,0.0))._1
            val stdev = reducedAttribute.getOrElse(key, (0.0, 0.0))._2
            val x = row(i).toDouble
            val p = (1.0/(stdev * scala.math.sqrt(2*scala.math.Pi))
              * scala.math.exp(-0.5*scala.math.pow((x-mean)/stdev,2)))
            probRowLabel *= p
          }
        }
        (label, probRowLabel)
      })
      val maxProb = probVector.maxBy(_._2)._1
      val correctLabel = ""+row(labelColumn)
      (correctLabel+","+maxProb+","+correctLabel.equals(maxProb),1)
    })

    val score = predict.reduceByKey(_+_)
    score.foreach(println)

  }
}