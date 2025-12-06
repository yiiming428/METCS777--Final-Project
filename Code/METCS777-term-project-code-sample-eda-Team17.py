from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when, expr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

spark = SparkSession.builder \
    .appName("Airbnb_EDA") \
    .getOrCreate()

df_la = spark.read.option("header", True).csv("/Users/yibingwang/Desktop/777/term project/la_cleaned.csv")
df_ny = spark.read.option("header", True).csv("/Users/yibingwang/Desktop/777/term project/ny_cleaned.csv")

# Missing value summary（LA vs NY）
def missing_table(df):
    return df.select([(count(when(col(c).isNull(), c)).alias(c)) for c in df.columns]).toPandas()

print("Missing - LA")
print(missing_table(df_la))

print("\nMissing - NY")
print(missing_table(df_ny))

#Descriptive statistics (key metrics comparison: count, min, mean, max, 25%, 50%, 75%)
#The goal is to compare LA vs NY listing structure – which city is more expensive,
# where prices are more concentrated, and where we see more extreme values.
desc_la = df_la.describe().toPandas()
desc_ny = df_ny.describe().toPandas()

print("LA Summary Stats:")
print(desc_la)

print("\nNY Summary Stats:")
print(desc_ny)

#Price distribution（Histogram + Log）
#Convert to Pandas (select price only)
la_price = df_la.select("price").dropna().toPandas()
ny_price = df_ny.select("price").dropna().toPandas()

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(la_price["price"], bins=50)
plt.title("LA Price Distribution")

plt.subplot(1,2,2)
plt.hist(ny_price["price"], bins=50)
plt.title("NY Price Distribution")

plt.show()

#Log-transformed price distribution

la_price["price"] = pd.to_numeric(la_price["price"], errors="coerce")
ny_price["price"] = pd.to_numeric(ny_price["price"], errors="coerce")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(np.log1p(la_price["price"].dropna()), bins=50)
plt.title("LA Log Price Distribution")

plt.subplot(1,2,2)
plt.hist(np.log1p(ny_price["price"].dropna()), bins=50)
plt.title("NY Log Price Distribution")

plt.tight_layout()
plt.show()

#Scatter Geo（Latitude vs Longitude, color = price）
#Check price range and extreme values
df_la.selectExpr(
    "max(price)", 
    "percentile(price, 0.95)", 
    "percentile(price, 0.99)"
).show()

df_ny.selectExpr(
    "max(price)", 
    "percentile(price, 0.95)", 
    "percentile(price, 0.99)"
).show()
#There are extreme price values observed in previous attempt 
#Keep only rows with 0 < price <= 2000
#For safety, convert price to double explicitly first
df_la_num = df_la.withColumn(
    "price_double",
    expr("try_cast(price as double)")
)

df_ny_num = df_ny.withColumn(
    "price_double",
    expr("try_cast(price as double)")
)

df_la_cap = df_la_num.filter(
    (col("price_double") > 0.0) & (col("price_double") <= 2000.0)
)

df_ny_cap = df_ny_num.filter(
    (col("price_double") > 0.0) & (col("price_double") <= 2000.0)
)

df_la_cap.selectExpr("min(price_double) as min_price", "max(price_double) as max_price").show()
df_ny_cap.selectExpr("min(price_double) as min_price", "max(price_double) as max_price").show()

print("LA rows after cap:", df_la_cap.count())
print("NY rows after cap:", df_ny_cap.count())

#plot
la_geo = df_la_cap.select("latitude", "longitude", "price_double").dropna().toPandas()
ny_geo = df_ny_cap.select("latitude", "longitude", "price_double").dropna().toPandas()

print("LA geo shape:", la_geo.shape)
print("NY geo shape:", ny_geo.shape)
print(la_geo.dtypes)
print(la_geo.head())
# In la_geo, all three columns are object type in Pandas, so convert to numeric

la_geo["latitude"] = pd.to_numeric(la_geo["latitude"], errors="coerce")
la_geo["longitude"] = pd.to_numeric(la_geo["longitude"], errors="coerce")
la_geo["price_double"] = pd.to_numeric(la_geo["price_double"], errors="coerce")

ny_geo["latitude"] = pd.to_numeric(ny_geo["latitude"], errors="coerce")
ny_geo["longitude"] = pd.to_numeric(ny_geo["longitude"], errors="coerce")
ny_geo["price_double"] = pd.to_numeric(ny_geo["price_double"], errors="coerce")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sc1 = plt.scatter(la_geo["longitude"], la_geo["latitude"],
                  c=la_geo["price_double"], s=5)
plt.colorbar(sc1, label="Price")
plt.title("LA Geo Price Map (price ≤ 2000)")
plt.xlabel("longitude")
plt.ylabel("latitude")

plt.subplot(1,2,2)
sc2 = plt.scatter(ny_geo["longitude"], ny_geo["latitude"],
                  c=ny_geo["price_double"], s=5)
plt.colorbar(sc2, label="Price")
plt.title("NY Geo Price Map (price ≤ 2000)")
plt.xlabel("longitude")
plt.ylabel("latitude")

plt.tight_layout()
plt.show()

#Numeric features vs Price scatter plot (with log-transformed price)

la_pd = df_la.select("minimum_nights","price").dropna().toPandas()
ny_pd = df_ny.select("minimum_nights","price").dropna().toPandas()
for df in (la_pd, ny_pd):
    df["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(la_pd["minimum_nights"], np.log1p(la_pd["price"]),
            s=8, alpha=0.5)
plt.title("LA: Minimum Nights vs Log(Price)")
plt.xlabel("Minimum Nights")
plt.ylabel("Log(Price + 1)")

plt.subplot(1,2,2)
plt.scatter(ny_pd["minimum_nights"], np.log1p(ny_pd["price"]),
            s=8, alpha=0.5)
plt.title("NY: Minimum Nights vs Log(Price)")
plt.xlabel("Minimum Nights")
plt.ylabel("Log(Price + 1)")

plt.tight_layout()
plt.show()

#Correlation Matrix（Heatmap）
numeric_cols = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
    "calculated_host_listings_count",
    "number_of_reviews_ltm",
    "latitude",
    "longitude"
]

la_corr = df_la.select(*numeric_cols).toPandas().corr()
ny_corr = df_ny.select(*numeric_cols).toPandas().corr()

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

#LA Heatmap
im1 = axes[0].imshow(la_corr, vmin=-1, vmax=1, cmap="coolwarm")
axes[0].set_title("LA Correlation Matrix", fontsize=18)
axes[0].set_xticks(range(len(numeric_cols)))
axes[0].set_yticks(range(len(numeric_cols)))
axes[0].set_xticklabels(numeric_cols, rotation=45, ha="right")
axes[0].set_yticklabels(numeric_cols)

cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8)
cbar1.set_label("Correlation", fontsize=12)

#NY Heatmap
im2 = axes[1].imshow(ny_corr, vmin=-1, vmax=1, cmap="coolwarm")
axes[1].set_title("NY Correlation Matrix", fontsize=18)
axes[1].set_xticks(range(len(numeric_cols)))
axes[1].set_yticks(range(len(numeric_cols)))
axes[1].set_xticklabels(numeric_cols, rotation=45, ha="right")
axes[1].set_yticklabels(numeric_cols)

cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8)
cbar2.set_label("Correlation", fontsize=12)

plt.tight_layout()
plt.show()