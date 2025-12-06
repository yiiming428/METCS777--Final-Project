from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row

DF_MODEL_PATH = "gs://cym111/FinalProject/df_model"
METRICS_OUTPUT_PATH = "gs://cym111/FinalProject/model_metrics"
IMPORTANCE_OUTPUT_PATH = "gs://cym111/FinalProject/feature_importance"


def train_and_eval(reg, train_df, test_df, city, model_name):
    # I fit the model on the training data and generate predictions on the test set.
    model = reg.fit(train_df)
    pred = model.transform(test_df)

    # I use a standard regression evaluator to compute RMSE, MAE, and R²,
    # since these give me a good sense of model performance from multiple angles.
    evaluator = RegressionEvaluator(
        labelCol="log_price",
        predictionCol="prediction"
    )

    rmse = evaluator.evaluate(pred, {evaluator.metricName: "rmse"})
    mae = evaluator.evaluate(pred, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(pred, {evaluator.metricName: "r2"})

    # I collect the evaluation metrics into a dictionary
    # so they can be saved later as part of my final output.
    metrics = {
        "city": city,
        "model": model_name,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    return model, metrics


def run_for_city(df_city, city_name):
    # I split the dataset into training (80%) and testing (20%) sets.
    # I fix the random seed so the split is reproducible.
    train_df, test_df = df_city.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.cache()
    test_df = test_df.cache()

    metrics_list = []
    importance_list = []

    # === 1. Linear Regression (baseline model) ===
    # I train a linear model first because it gives me a simple, interpretable baseline.
    lr = LinearRegression(
        featuresCol="features",
        labelCol="log_price",
        maxIter=50,
        regParam=0.1,
        elasticNetParam=0.5,
    )
    lr_model, m_lr = train_and_eval(lr, train_df, test_df, city_name, "LinearRegression")
    metrics_list.append(m_lr)

    # === 2. Random Forest (nonlinear, stronger baseline) ===
    # I use a reasonably sized forest to balance performance and runtime.
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="log_price",
        numTrees=100,
        maxDepth=8,
        maxBins=32,
        seed=42,
    )
    rf_model, m_rf = train_and_eval(rf, train_df, test_df, city_name, "RandomForest")
    metrics_list.append(m_rf)

    # After training the RF model, I extract and store feature importances.
    feature_importances = rf_model.featureImportances.toArray().tolist()
    for idx, score in enumerate(feature_importances):
        importance_list.append(
            {
                "city": city_name,
                "model": "RandomForest",
                "feature_index": int(idx),
                "importance": float(score),
            }
        )

    # === 3. Gradient Boosted Trees (typically strong but heavier) ===
    # I wrap this in a try block because GBT can occasionally fail on small datasets.
    try:
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="log_price",
            maxIter=30,
            maxDepth=4,
            maxBins=32,
            stepSize=0.1,
            subsamplingRate=0.8,
            seed=42,
        )
        gbt_model, m_gbt = train_and_eval(gbt, train_df, test_df, city_name, "GBT")
        metrics_list.append(m_gbt)

        # Again, I record feature importances for analysis.
        gbt_importances = gbt_model.featureImportances.toArray().tolist()
        for idx, score in enumerate(gbt_importances):
            importance_list.append(
                {
                    "city": city_name,
                    "model": "GBT",
                    "feature_index": int(idx),
                    "importance": float(score),
                }
            )
    except Exception as e:
        # I print a warning instead of crashing the job if GBT fails.
        print(f"WARNING: GBT training failed for {city_name}: {e}")

    return metrics_list, importance_list


def main():
    # I begin by creating the Spark session for the modeling stage.
    spark = (
        SparkSession.builder
        .appName("Airbnb-Modeling-LA-vs-NY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # I load the modeling dataset generated during the feature engineering step.
    print(f"Reading df_model from: {DF_MODEL_PATH}")
    df_model = spark.read.parquet(DF_MODEL_PATH).cache()

    # I split the data by city so each city has its own model comparison.
    df_la = df_model.filter(df_model.city == "LA")
    df_ny = df_model.filter(df_model.city == "NY")

    all_metrics = []
    all_importances = []

    # === Train models for LA ===
    print("Training models for LA...")
    m_la, imp_la = run_for_city(df_la, "LA")
    all_metrics.extend(m_la)
    all_importances.extend(imp_la)

    # === Train models for NY ===
    print("Training models for NY...")
    m_ny, imp_ny = run_for_city(df_ny, "NY")
    all_metrics.extend(m_ny)
    all_importances.extend(imp_ny)

    # === Save metrics to GCS ===
    metrics_df = spark.createDataFrame([Row(**m) for m in all_metrics])
    print("Model metrics:")
    metrics_df.show(truncate=False)

    metrics_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
        METRICS_OUTPUT_PATH
    )

    # === Save feature importances to GCS ===
    importance_df = spark.createDataFrame([Row(**r) for r in all_importances])
    print("Feature importance (top few rows):")
    importance_df.show(20, truncate=False)

    importance_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
        IMPORTANCE_OUTPUT_PATH
    )

    spark.stop()


if __name__ == "__main__":
    # I call main() so the full modeling pipeline runs when this script executes.
    main()
