from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# ================== Config Section ==================
# I keep all GCS paths parameterized so that I only need to change BUCKET_BASE
# when switching environment or bucket names.
BUCKET_BASE = "gs://cym111/FinalProject"

LA_CLEAN_PATH = f"{BUCKET_BASE}/la_cleaned.csv"
NY_CLEAN_PATH = f"{BUCKET_BASE}/ny_cleaned.csv"

# This will be the output folder containing the modeling-ready parquet files.
DF_MODEL_OUTPUT_PATH = f"{BUCKET_BASE}/df_model"
# ====================================================


def main():
    # I start by creating a Spark session, which serves as the entry point
    # for running distributed jobs on Dataproc.
    spark = (
        SparkSession.builder
        .appName("airbnb_feature_engineering")
        .getOrCreate()
    )

    # === Load cleaned datasets for LA and NY ===
    # I read the cleaned LA file, allow schema inference, and attach a "city" column.
    la_df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(LA_CLEAN_PATH)
    ).withColumn("city", F.lit("LA"))

    # Same process for the NY dataset.
    ny_df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(NY_CLEAN_PATH)
    ).withColumn("city", F.lit("NY"))

    # Before merging the two datasets, I make sure they share the same columns.
    # I take the intersection so the union operation is compatible.
    common_cols = list(set(la_df.columns).intersection(set(ny_df.columns)))
    la_df = la_df.select(common_cols)
    ny_df = ny_df.select(common_cols)

    # Now I combine the two datasets into a single dataframe.
    df = la_df.unionByName(ny_df)

    # === Basic validation and filtering ===
    # I define a set of columns that must exist for my later modeling steps.
    required_cols = [
        "price",
        "latitude",
        "longitude",
        "room_type",
        "neighbourhood_group",
    ]

    # If any required field is missing, I prefer to fail early with an error.
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing necessary fields: {missing}")

    # I filter out rows with null or non-positive prices because
    # the model uses a log transformation, which requires positive values.
    df = df.filter(F.col("price").isNotNull() & (F.col("price") > 0))

    # === Feature: log_price ===
    # I create the log-transformed price, which usually helps stabilize variance.
    df = df.withColumn("log_price", F.log1p(F.col("price")))

    # === Feature: distance to city center ===
    # I compute a simple Euclidean distance from each listing to its city center.
    # Although not a perfect geospatial metric, it serves as a useful proxy for location.
    df = df.withColumn(
        "dist_to_center",
        F.when(
            F.col("city") == "LA",
            F.sqrt(
                (F.col("latitude") - F.lit(34.0522)) ** 2
                + (F.col("longitude") - F.lit(-118.2437)) ** 2
            ),
        ).otherwise(
            F.sqrt(
                (F.col("latitude") - F.lit(40.7128)) ** 2
                + (F.col("longitude") - F.lit(-74.0060)) ** 2
            )
        ),
    )

    # === Prepare numeric features ===
    # These are the numeric variables I plan to include in the model.
    candidate_numeric = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "availability_365",
        "calculated_host_listings_count",
        "number_of_reviews_ltm",
        "dist_to_center",
        "latitude",
        "longitude",
    ]

    # I keep only the ones that actually exist in the dataset.
    numeric_features = [c for c in candidate_numeric if c in df.columns]

    if not numeric_features:
        raise ValueError("No numeric features found. Please verify dataset columns.")

    # === Prepare categorical features ===
    # I include room_type and neighbourhood_group as categorical predictors.
    categorical_cols = []
    for c in ["room_type", "neighbourhood_group"]:
        if c in df.columns:
            categorical_cols.append(c)

    if not categorical_cols:
        raise ValueError("No categorical features found. Expected room_type or neighbourhood_group.")

    # === Drop rows with missing values in important fields ===
    # I remove rows that have missing values in any of the required modeling columns.
    df = df.na.drop(subset=numeric_features + categorical_cols + ["log_price"])

    # === Build feature engineering pipeline ===
    # First, I convert each categorical column into category indices.
    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_idx",
            handleInvalid="keep",
        )
        for col in categorical_cols
    ]

    # Next, I one-hot encode each indexed categorical column.
    encoders = [
        OneHotEncoder(
            inputCol=f"{col}_idx",
            outputCol=f"{col}_ohe",
            dropLast=True,
        )
        for col in categorical_cols
    ]

    # I gather the names of all encoded feature columns.
    ohe_cols = [f"{col}_ohe" for col in categorical_cols]

    # Finally, I combine numeric and OHE columns into a single vector.
    assembler = VectorAssembler(
        inputCols=numeric_features + ohe_cols,
        outputCol="features",
        handleInvalid="keep",
    )

    # I package all steps into one pipeline so they are applied consistently.
    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    # Fit the feature pipeline on the full dataset.
    model = pipeline.fit(df)

    # Apply the transformations to produce the final modeling dataset.
    fe_df = model.transform(df)

    # I keep only the columns needed for modeling.
    df_model = fe_df.select("city", "log_price", "features")

    # === Save the final dataframe to GCS ===
    # I write df_model to parquet so that later ML scripts can read it directly.
    df_model.write.mode("overwrite").parquet(DF_MODEL_OUTPUT_PATH)

    print("df_model successfully written to:", DF_MODEL_OUTPUT_PATH)
    print("Row count:", df_model.count())

    spark.stop()


if __name__ == "__main__":
    # I call main() so the entire pipeline runs when this script executes.
    main()
