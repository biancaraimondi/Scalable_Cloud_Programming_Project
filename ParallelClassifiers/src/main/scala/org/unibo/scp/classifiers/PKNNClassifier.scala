package org.unibo.scp.classifiers

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

class PKNNClassifier(val k: Int = 5, val sparkSession: SparkSession) extends Serializable {
  var trainSet: RDD[(Int, Array[Double], Int)] = _

  def train(trainSet: RDD[(Int, Array[Double], Int)]): Unit = {
    this.trainSet = trainSet
    println("Training set size: " + trainSet.count())
  }

  def predict(query: Array[(Int, Array[Double])]): RDD[(Int, Int)] = {
    trainSet
      .mapPartitions(p => {
        val localPartition = p.toArray
        query.map { case (id, sample) =>
          id -> getKNeighbours(sample, localPartition, k)
        }.iterator
      })
      .reduceByKey(Array(_, _).flatten.sortBy(_._1).take(k))
      .map { case (id, tuples) =>
        id -> tuples
          .groupBy(_._2)
          .map { case (i, tuples) => i -> tuples.length }
          .maxBy(_._2)
          ._1
      }
  }

  def score(yTest: Array[(Int, Int)], yPred: Array[(Int, Int)]) = {
    val n = yTest.length
    val n2 = yPred.length

    assert(n == n2)

    val y1 = yTest.sortBy(_._1)
    val y2 = yPred.sortBy(_._1)

    val confusionMap = sparkSession.sparkContext.parallelize(
      y1.zip(y2)
        .map { case ((id1, l1), (id2, l2)) =>
          assert(id1 == id2)
          (l1 == l2, 1)
        }
    ).reduceByKey(_ + _).collectAsMap()

    confusionMap.getOrElse(true, 0).doubleValue / n.doubleValue
  }

  private def getKNeighbours(q: Array[Double],
                             trainingSamples: Array[(Int, Array[Double], Int)], k: Int)
  : Array[(Double, Int)] = {

    trainingSamples
      .map { case (_, sample, label) =>
        (KNNUtils.euclideanDistance(q, sample), label)
      }
      .sortBy(_._1)
      .take(k)
  }
}
