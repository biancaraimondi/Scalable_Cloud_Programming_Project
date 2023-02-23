import org.apache.spark.sql.{SparkSession}

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

    val labelColumn = 0
    val colName = rdd.first()
//    colName.foreach(println)
    val data = rdd.filter(line => !(line.sameElements(colName)))

    //get every possible label
    val allLabels = data.map(row => row(labelColumn)).distinct().collect()
//    allLabels.foreach(println)


    //split into test and train set
    val Array(trainSet, testSet) = data.randomSplit(Array(0.7,0.3))

    /**MAP:
     *in each partition of the training set we need to analyse each row and add a tuple (key, 1)
     * the key will be label or a string composed by variable name, variable value and label*/
    //zero frequency problem not handled
    //continuous data (gaussian?) not handled
    val mapped = trainSet.flatMap(row =>{
      var seq = Seq.empty[(String, Int)]
      seq = seq :+ (""+row(labelColumn), 1)
      //for (i<- to (row.length-1))
      for (i<-0 to 4){
        if(i != labelColumn){
          val attibuteName = colName(i)
          val attributeValue = row(i)
          val key =attibuteName + "-" + attributeValue + "-" + row(labelColumn)
          seq = seq :+ (key, 1)
        }
      }
      seq
    })

    /**REDUCE
     * simply reduce by key summing over the value with the same key*/
    val reduced = mapped.reduceByKey(_+_).collectAsMap()
    reduced.foreach(println)

    /**PREDICT*/
    val predict = testSet.map(row => {
      //foreach label of allLabels
      val probVector = allLabels.map(label => {
        val currentLabel = label
        val freqCurrentLabel:Int = reduced.getOrElse(currentLabel,0)
        var probRowLabel:Float = freqCurrentLabel
        println("cf: "+probRowLabel)
        for (i <- 0 until 4){
          if (i != labelColumn){
            val attributeName = colName(i)
            val attributeValue = row(i)
            val key = attributeName+"-"+attributeValue+"-"+currentLabel
            val freqEvent:Int = reduced.getOrElse(key,0)
            val probEvent:Float = freqEvent.toFloat/freqCurrentLabel

            println("k: "+probEvent)
            probRowLabel *= probEvent
          }
        }
        (label, probRowLabel)
      })
      probVector.foreach(println)
      val maxProb = probVector.maxBy(_._2)._1
      val correctLabel = ""+row(labelColumn)
      (correctLabel+","+maxProb+","+correctLabel.equals(maxProb),1)
    })

    val score = predict.reduceByKey(_+_)
    score.foreach(println)
  }
}