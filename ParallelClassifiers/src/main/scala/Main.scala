import classifiers.{GaussianNB, MultinomialNB}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("ParallelClassifier")
      .getOrCreate()

    //    val rddFromFile = sc.textFile("src/main/data/data.csv")
    val rddFromFile = spark.sparkContext.textFile("src/main/data/wine.csv")
    val rdd = rddFromFile.map(f => {
      f.split(";")
    })

    val colName = rdd.first()
    val rddWOHeader = rdd.filter(line => !line.sameElements(colName))

    val rddLabeled: RDD[LabeledPoint] = rddWOHeader.map(row =>  LabeledPoint(
      row.last.toDouble,
      Vectors.dense(row.init.map(_.toDouble))
    ))
    val Array(trainSet, testSet) = rddLabeled.randomSplit(Array(0.7,0.3))

    /** GAUSSIAN NAIVEBAYES MODEL */
    val gaussianNB = new GaussianNB(spark)
    gaussianNB.colName = colName

    gaussianNB.train(trainSet)

    val accuracyGNB = gaussianNB.accuracy(testSet)
    println(s"GNB accuracy: $accuracyGNB")

    /** DECISION TREE MLLIB IMPLEMENTATION */
    val numClasses = 2
    val categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val modelDT = DecisionTree.trainClassifier(trainSet, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    val labelAndPreds = testSet.map { point =>
      val prediction = modelDT.predict(point.features)
      (point.label, prediction)
    }

    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testSet.count()
    println("DT ACCURACY ="+(1-testErr))

    while(true){

    }
  }
}