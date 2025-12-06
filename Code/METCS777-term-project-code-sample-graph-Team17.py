import pandas as pd
import matplotlib.pyplot as plt

# I load the feature-importance CSV that was generated from the modeling step.
# If the column names differ, I can always inspect them by printing fi.columns.
fi = pd.read_csv("feature_importance.csv")


def plot_feature_importance(data, city, model, outfile):
    """
    For a given city and model, I filter the feature-importance rows,
    sort them by importance, and generate a clean horizontal bar plot.
    """

    # I select only the rows that match the city–model pair.
    sub = data[(data["city"] == city) & (data["model"] == model)].copy()

    # I sort the features so the most important ones appear at the top of the chart.
    sub = sub.sort_values("importance", ascending=False)

    # I make a horizontal bar plot because it's more readable with many features.
    plt.figure(figsize=(6, 4))
    plt.barh(sub["feature_index"].astype(str), sub["importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Index")
    plt.title(f"{city} - {model} Feature Importance")

    # I invert the y-axis so the highest-importance feature appears at the top.
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# I generate the four importance charts:
#   - LA Random Forest
#   - LA GBT
#   - NY Random Forest
#   - NY GBT
plot_feature_importance(fi, "LA", "RandomForest", "LA_RF_importance.png")
plot_feature_importance(fi, "LA", "GBT",          "LA_GBT_importance.png")
plot_feature_importance(fi, "NY", "RandomForest", "NY_RF_importance.png")
plot_feature_importance(fi, "NY", "GBT",          "NY_GBT_importance.png")
