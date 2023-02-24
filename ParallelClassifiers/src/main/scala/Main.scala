import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .master("local[1]")
      .appName("ParallelClassifier")
      .getOrCreate()

    //    val rddFromFile = sc.textFile("src/main/data/data.csv")
    val rddFromFile = spark.sparkContext.textFile("src/main/data/wine.csv")
    val rdd = rddFromFile.map(f => {
      f.split(";")
    })

    val gaussianNB = new GaussianNB(spark)

    gaussianNB.colName = rdd.first()

    val Array(trainSet, testSet) = rdd.randomSplit(Array(0.7,0.3))

    gaussianNB.train(trainSet)

    val score = gaussianNB.score(testSet)

    score.foreach(println)
  }
}