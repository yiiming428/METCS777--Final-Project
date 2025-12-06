from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr

spark = SparkSession.builder \
    .appName("Airbnb_Cleaning") \
    .getOrCreate()

df_la_raw = spark.read.option("header", True).option("inferSchema", True).csv("/Users/yibingwang/Desktop/777/term project/sample_la.csv")
df_ny_raw = spark.read.option("header", True).option("inferSchema", True).csv("/Users/yibingwang/Desktop/777/term project/sample_ny.csv")

df_la_raw.printSchema()
df_la_raw.show(5)

df_ny_raw.printSchema()
df_ny_raw.show(5)

#remove rows where ID is missing (detect and delete the rows where ID is not numeric)
df_la_base = df_la_raw.filter(
    col("id").isNotNull() &
    (col("id") != "") &
    col("id").rlike("^[0-9]+$")
)

df_ny_base = df_ny_raw.filter(
    col("id").isNotNull() &
    (col("id") != "") &
    col("id").rlike("^[0-9]+$")
)

#According to schema，many ny fields are read as strings. 
#Casting to double is required for further numeric processing
#df_ny = (
#    df_ny_raw
#        .withColumn("price", regexp_replace(col("price"), "[$,]", "").cast("double"))
#        .withColumn("latitude", col("latitude").cast("double"))
#        .withColumn("longitude", col("longitude").cast("double"))
#        .withColumn("minimum_nights", col("minimum_nights").cast("double"))
#        .withColumn("number_of_reviews", col("number_of_reviews").cast("double"))
#        .withColumn("reviews_per_month", col("reviews_per_month").cast("double"))
#        .withColumn("calculated_host_listings_count", col("calculated_host_listings_count").cast("double")))
#some cols cannot be directly cast to double. There will be error messages like: The value 'Private room' ... cannot be cast to "DOUBLE"
#Therefore, we switch to using try_cast to safely convert values without throwing errors.
df_ny = (
    df_ny_base
        .withColumn("price", expr("try_cast(regexp_replace(price, '[$,]', '') as double)"))
        .withColumn("latitude", expr("try_cast(latitude as double)"))
        .withColumn("longitude", expr("try_cast(longitude as double)"))
        .withColumn("minimum_nights", expr("try_cast(minimum_nights as double)"))
        .withColumn("number_of_reviews", expr("try_cast(number_of_reviews as double)"))
        .withColumn("reviews_per_month", expr("try_cast(reviews_per_month as double)"))
        .withColumn("calculated_host_listings_count", expr("try_cast(calculated_host_listings_count as double)"))
        .withColumn("availability_365", expr("try_cast(availability_365 as double)"))
        .withColumn("number_of_reviews_ltm", expr("try_cast(number_of_reviews_ltm as double)"))
)

df_la = (
    df_la_base
        .withColumn("price", expr("try_cast(price as double)"))
        .withColumn("latitude", expr("try_cast(latitude as double)"))
        .withColumn("longitude", expr("try_cast(longitude as double)"))
        .withColumn("minimum_nights", expr("try_cast(minimum_nights as double)"))
        .withColumn("number_of_reviews", expr("try_cast(number_of_reviews as double)"))
        .withColumn("reviews_per_month", expr("try_cast(reviews_per_month as double)"))
        .withColumn("calculated_host_listings_count", expr("try_cast(calculated_host_listings_count as double)"))
        .withColumn("availability_365", expr("try_cast(availability_365 as double)"))
        .withColumn("number_of_reviews_ltm", expr("try_cast(number_of_reviews_ltm as double)"))
)
#Convert last_review to date
df_ny = df_ny.withColumn("last_review", expr("try_cast(last_review as date)"))
df_la = df_la.withColumn("last_review", expr("try_cast(last_review as date)"))

#Fill missing review related fields
df_ny = df_ny.fillna({
    "reviews_per_month": 0.0,
    "number_of_reviews_ltm": 0.0
})
df_la = df_la.fillna({
    "reviews_per_month": 0.0,
    "number_of_reviews_ltm": 0.0
})

#Remove listings without price
df_la = df_la.filter(col("price").isNotNull())
df_ny = df_ny.filter(col("price").isNotNull())
#save clean data
df_la.write.mode("overwrite").option("header", True).csv("/Users/yibingwang/Desktop/777/term project/la_cleaned")
df_ny.write.mode("overwrite").option("header", True).csv("/Users/yibingwang/Desktop/777/term project/ny_cleaned")