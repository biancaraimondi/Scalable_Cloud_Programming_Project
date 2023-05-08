package org.unibo.scp

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.unibo.scp.classifiers._

object GNBExecutor {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("GNB App")
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
  }
}