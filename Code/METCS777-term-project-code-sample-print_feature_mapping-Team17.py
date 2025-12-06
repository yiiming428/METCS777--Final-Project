from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    OneHotEncoderModel,
    VectorAssembler,
)
from pyspark.ml import Pipeline

# ================== Config Section ==================
# I keep the base bucket path here so I can easily switch environments
# without editing multiple parts of the script.
BUCKET_BASE = "gs://cym111/FinalProject"

LA_CLEAN_PATH = f"{BUCKET_BASE}/la_cleaned.csv"
NY_CLEAN_PATH = f"{BUCKET_BASE}/ny_cleaned.csv"
# ====================================================


def main():
    # I create a Spark session dedicated to printing the feature-index mapping.
    spark = (
        SparkSession.builder
        .appName("print_feature_mapping_local_pipeline")
        .getOrCreate()
    )

    print("\n=== Load cleaned LA / NY data ===")
    # I load the LA dataset, infer its schema, and tag it with a city indicator.
    la_df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(LA_CLEAN_PATH)
    ).withColumn("city", F.lit("LA"))

    # Same process for NY.
    ny_df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(NY_CLEAN_PATH)
    ).withColumn("city", F.lit("NY"))

    # Before merging the two datasets, I keep only columns that exist in both.
    common_cols = list(set(la_df.columns).intersection(set(ny_df.columns)))
    la_df = la_df.select(common_cols)
    ny_df = ny_df.select(common_cols)

    df = la_df.unionByName(ny_df)

    # === Initial feature engineering steps (must match the real pipeline) ===
    required_cols = [
        "price",
        "latitude",
        "longitude",
        "room_type",
        "neighbourhood_group",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # I drop rows with missing or invalid prices because they break log transformation.
    df = df.filter(F.col("price").isNotNull() & (F.col("price") > 0))

    # I generate the log-price target variable.
    df = df.withColumn("log_price", F.log1p(F.col("price")))

    # I compute the distance to the city center (LA or NY),
    # using a simple Euclidean formula as a reasonable location proxy.
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

    # Numeric features I expect to include.
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
    numeric_features = [c for c in candidate_numeric if c in df.columns]
    if not numeric_features:
        raise ValueError("No numeric features found.")

    # Categorical features I plan to index and encode.
    categorical_cols = []
    for c in ["room_type", "neighbourhood_group"]:
        if c in df.columns:
            categorical_cols.append(c)
    if not categorical_cols:
        raise ValueError("No categorical features found.")

    # I remove rows with missing feature values, to ensure consistency.
    df = df.na.drop(subset=numeric_features + categorical_cols + ["log_price"])

    # I index categorical fields so they can be one-hot encoded.
    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_idx",
            handleInvalid="keep",
        )
        for col in categorical_cols
    ]

    # Then I apply one-hot encoding, dropping the last category to avoid redundancy.
    encoders = [
        OneHotEncoder(
            inputCol=f"{col}_idx",
            outputCol=f"{col}_ohe",
            dropLast=True,
        )
        for col in categorical_cols
    ]

    ohe_cols = [f"{col}_ohe" for col in categorical_cols]

    # I assemble numeric and encoded categorical features into a single vector.
    assembler = VectorAssembler(
        inputCols=numeric_features + ohe_cols,
        outputCol="features",
        handleInvalid="keep",
    )

    # I construct the pipeline exactly as used in the real modeling step.
    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    print("\n=== Fitting pipeline (for mapping only) ===")
    # I fit the pipeline so I can inspect its internal structures.
    model = pipeline.fit(df)

    # === Print the feature index mapping ===
    print("\n==============================")
    print(" FEATURE INDEX MAPPING TABLE ")
    print("==============================")

    # I collect the one-hot encoder models from the fitted pipeline.
    encoder_models = [
        s for s in model.stages if isinstance(s, OneHotEncoderModel)
    ]

    # I map each categorical variable to the dimensionality of its OHE encoding.
    col2dim = {}
    for col in categorical_cols:
        enc = next(
            e for e in encoder_models
            if e.getInputCol() == f"{col}_idx"
        )
        # categorySizes gives the number of original categories.
        # With dropLast=True, the encoded dimension is (n_categories - 1).
        n_cat = enc.categorySizes[0]
        dim = n_cat - 1
        col2dim[col] = dim

    feature_index = 0

    print("\n--- Numeric Features ---")
    for col in numeric_features:
        print(f"{feature_index}: {col}")
        feature_index += 1

    print("\n--- Categorical OHE Features ---")
    for col in categorical_cols:
        dim = col2dim[col]
        ohe_name = f"{col}_ohe"
        print(f"\n{ohe_name} (dim = {dim}):")
        for j in range(dim):
            print(f"{feature_index}: {ohe_name}_{j}")
            feature_index += 1

    print("\nTotal feature dimension =", feature_index)

    spark.stop()
    print("\nDone, mapping printed above.\n")


if __name__ == "__main__":
    main()
