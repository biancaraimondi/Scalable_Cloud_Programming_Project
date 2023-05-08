package org.unibo.scp

import org.apache.spark.sql._
import org.unibo.scp.classifiers._

object KNNExecutor {
  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      println(s"Error starting the program. Usage: <program-name> <train-file-path> <test-file-path>")
      sys.exit(1)
    }
    val trainPath = args(0)
    val testPath = args(1)
    // val usedCores = args(1).toInt
    val usedCores = Runtime.getRuntime.availableProcessors()
    val spark = SparkSession.builder()
      .appName("Parallel KNN App")
      .master(s"local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    // val df = spark.sparkContext.textFile()

    val df = spark.read
      .option("header", "true")
      .csv(trainPath)

    /*val trainSize = if (trainPath.contains("SUSY")) {
      30000
    } else {
      (df.count() * 0.8).toInt
    }

    val testSize = (trainSize * 0.25).toInt

    println(s"Train size: $trainSize")
    println(s"Test size: $testSize")*/

    val trainSet = spark.read
      .option("header", "true")
      .csv(trainPath)
      // .limit(trainSize)
      // .sample(0.3)
      .drop("date")
      .rdd
      .map(formatRow)
      .repartition(usedCores)
      // .persist()

    val testSet = spark.read
      .option("header", "true")
      .csv(testPath)
      // .sample(0.25)
      // .limit(testSize)
      .drop("date")
      .collect()
      .map(formatRow)

    val trainSize = trainSet.count()
    val testSize = testSet.size

    println(s"Using $usedCores cores.")
    println(s"Train size: $trainSize.")
    println(s"Test size: $testSize.")

    // val testSet = df
    //   // .where(df("id") > trainSize)
    //   .limit(testSize)
    //   .drop("date")
    //   .collect()
    //   .map(formatRow)

    // setting k
    val k = 5

    val classifier = new PKNNClassifier(k, spark)
    classifier.train(trainSet)

    val xTest = testSet.map { case (id, features, _) => id -> features }
    val yTest = testSet.map { case (id, _, label) => id -> label }

    val t0 = System.nanoTime()
    val yPred = classifier
      .predict(xTest)
      .collect()
      .sortBy(_._1)
    val t1 = System.nanoTime()

    val timeInSeconds = (t1 - t0) / 1e9
    println(s"Elapsed time: $timeInSeconds")

    val accuracy = classifier.score(yTest, yPred)

    println(s"Accuracy: $accuracy")
  }

  private def formatRow(row: Row) = {
    val x = row.toSeq.toArray
    val sample = x.slice(1, x.length - 1).map(_.toString.toDouble)
    val label = x.last.toString.toInt

    (x(0).toString.toInt, sample, label)
  }
}
