package classifiers
object KNNUtils {
  def euclideanDistance(x: Array[Double], y: Array[Double]): Double = {
    math.sqrt(x.zip(y).map { case (a, b) => math.pow(a - b, 2) }.sum)
  }
}
