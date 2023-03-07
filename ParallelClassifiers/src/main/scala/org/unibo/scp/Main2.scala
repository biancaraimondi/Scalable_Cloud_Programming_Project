package org.unibo.scp

import org.apache.spark.sql._
import classifiers._

object Main2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("KNNClassifier")
      .master("local[*]")
      .getOrCreate()

    val filepath = "src/main/data/occupancy_train.csv"
    val df = spark.read.option("header", "true")
      .csv(filepath)

    val trainSize = df.count() * 0.8

    val trainSet = df
      .limit(trainSize.toInt)
      .drop("date")
      .rdd
      .map(formatRow)
      .repartition(4)

    val testSet = df
      .where(df("id") > trainSize)
      .drop("date")
      .collect()
      .map(formatRow)

    // setting k
    val k = 5

    val classifier = new PKNNClassifier(k, spark)
    classifier.train(trainSet)

    val xTest = testSet.map { case (id, features, _) => id -> features }
    val yTest = testSet.map { case (id, _, label) => id -> label }

    val yPred = classifier
      .predict(xTest)
      .collect()
      .sortBy(_._1)

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
