package classifiera

class KNNClassifier(val k: Int = 5) {
  private var xTrain: Array[Array[Double]] = _
  private var yTrain: Array[Int] = _

  def train(x: Array[Array[Double]], y: Array[Int]): Unit = {
    xTrain = x
    yTrain = y
  }

  def predict(query: Array[Double]): Int = {
    xTrain.zip(yTrain)
      .map(row => {
        val distance = KNNUtils.euclideanDistance(row._1, query)
        (distance, row._2)
      })
      .sortBy(_._1)
      .take(k)
      .groupBy(_._2)
      .map(row => (row._1, row._2.length))
      .maxBy(_._2)._1
  }

  def score(xTest: Array[Array[Double]], yTest: Array[Int]): Double = {
    val yPred = xTest.map(predict)
    yPred.zip(yTest)
      .map(x => if (x._1 == x._2) 1 else 0)
      .sum.toDouble / yTest.length
  }
}
