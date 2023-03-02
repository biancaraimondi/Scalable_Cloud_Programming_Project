import org.apache.spark.sql.SparkSession

object OccupancyPreprocessing {
  def main(args: Array[String]): Unit = {

    // Create a local SparkSession
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Preprocessing")
      .getOrCreate()

    // get data from "./data/winequality-red.csv" and "./data/winequality-white.csv"
    val occupancy = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/data/occupancy.csv")

    // remove id and data columns
    val occupancy_drop = occupancy.drop("id", "date")

    // save the merged dataframe to "./data/wine.csv"
    occupancy_drop.write
      .option("header", "true")
      .option("inferSchema", "true")
      .mode("overwrite")
      .option("delimiter", ",")
      .csv("src/main/data/occupancy")
  }
}