package org.unibo.scp.utils

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object RF_SVM {
  def main(args: Array[String]): Unit = {

    // Create a local SparkSession
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("RandomForestClassifierExample")
      .getOrCreate()

    // create an RDD from "data/wine.csv" file
    val rddFromFile = spark
      .sparkContext
      .textFile("src/main/data/wine.csv")

    val rdd = rddFromFile.map ( f => {
      f.split(";")
    })

    val colName = rdd.first()
    val rddWOHeader = rdd.filter(line => !line.sameElements(colName))

    val rddLabeled: RDD[LabeledPoint] = rddWOHeader.map(row => LabeledPoint(
      row.last.toDouble,
      Vectors.dense(row.init.map(_.toDouble))
    ))
    val Array(trainSet, testSet) = rddLabeled.randomSplit(Array(0.7, 0.3))


    /** RANDOM FOREST MLLIB IMPLEMENTATION */
    val numClasses = 2
    val categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32
    val numTrees = 6 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val seed = 12345

    val modelRF = RandomForest.trainClassifier(trainSet, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed)

    val labelAndPredsRF = testSet.map { point =>
      val prediction = modelRF.predict(point.features)
      (point.label, prediction)
    }

    val testErrRF = labelAndPredsRF.filter(r => r._1 != r._2).count().toDouble / testSet.count()
    println("RF ACCURACY =" + (1 - testErrRF))


    /** SVM MLLIB IMPLEMENTATION */
    val numIterations = 100
    val modelSVM = SVMWithSGD.train(trainSet, numIterations)

    val labelAndPredsSVM = testSet.map { point =>
      val prediction = modelSVM.predict(point.features)
      (point.label, prediction)
    }

    val testErrSVM = labelAndPredsSVM.filter(r => r._1 != r._2).count().toDouble / testSet.count()
    println("SVM ACCURACY =" + (1 - testErrSVM))

  }
}