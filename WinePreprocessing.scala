import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.lit

class WinePreprocessing {
  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder()
      .master("local[1]")
      .appName("SparkByExamples.com")
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

    // remove " and / from column names
    var wine_cleaned = wine.columns.foldLeft(wine)((df, c) => df.withColumnRenamed(c, c.replaceAll("\"", "")))

    // write the merged dataframe to a CSV file
    wine_cleaned = wine_cleaned.coalesce(1)
    wine_cleaned = wine_cleaned.columns.foldLeft(wine_cleaned)((df, c) => df.withColumnRenamed(c, c.replaceAll(" ", "")))


    wine_cleaned.write
      .option("header", "true") // write header row
      .option("delimiter", ",") // use comma as separator
      .mode("overwrite") // overwrite the file if it already exists
      .csv("src/main/data/wine.csv")

  }
}
