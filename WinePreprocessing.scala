import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.lit

object WinePreprocessing {
  def main(args: Array[String]): Unit = {

    // Create a local SparkSession
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Preprocessing")
      .getOrCreate()

    // get data from "./data/winequality-red.csv" and "./data/winequality-white.csv"
    val red_wine = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/data/winequality-red.csv")
    // add int column to red wine dataframe to indicate red wine
    val red_wine_with_color = red_wine.withColumn("color", lit(1))

    val white_wine = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/data/winequality-white.csv")
    // add int column to white wine dataframe to indicate white wine
    val white_wine_with_color = white_wine.withColumn("color", lit(0))

    // merge the two dataframes
    val wine = red_wine_with_color.union(white_wine_with_color)

    // write the merged dataframe to a CSV file
    val wine_coalesce = wine.coalesce(1)

    // save the merged dataframe to "./data/wine.csv"
    wine_coalesce.write
      .option("header", "true")
      .option("inferSchema", "true")
      .mode("overwrite")
      .option("delimiter", ",")
      .csv("src/main/data/wine.csv")
  }
}