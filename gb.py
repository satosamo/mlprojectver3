import pandas as pd
import numpy as np
import json
import re
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
import seaborn as sns

movies_df = pd.read_csv("sample_data/tmdb_5000_movies.csv")

movies_df = movies_df[(movies_df["revenue"] > 0) & (movies_df["revenue"] <= 100_000_000)]

# budget and revenue to billions
movies_df["revenue"] = movies_df["revenue"] / 1_000_000_000
movies_df["budget"] = movies_df["budget"] / 1_000_000_000

# Log
movies_df["log_revenue"] = np.log1p(movies_df["revenue"])

movies_df.dropna(subset=["budget", "revenue", "genres", "runtime", "release_date", "overview"], inplace=True)


def extract_main_genre(genre_json):
    try:
        genres = json.loads(genre_json.replace("'", "\""))  # fix quotes if necessary
        return genres[0]["name"] if genres else None
    except (json.JSONDecodeError, TypeError):
        return None

movies_df["genre_main"] = movies_df["genres"].apply(extract_main_genre)
movies_df.dropna(subset=["genre_main"], inplace=True)

# One-hot
genre_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
genre_encoded = genre_encoder.fit_transform(movies_df[["genre_main"]])
genre_encoded_df = pd.DataFrame(genre_encoded, columns=genre_encoder.get_feature_names_out(["genre_main"]))
movies_df = pd.concat([movies_df.reset_index(drop=True), genre_encoded_df.reset_index(drop=True)], axis=1)

movies_df["release_date"] = pd.to_datetime(movies_df["release_date"], errors="coerce")
movies_df.dropna(subset=["release_date"], inplace=True)

movies_df["release_year"] = movies_df["release_date"].dt.year
movies_df["release_month"] = movies_df["release_date"].dt.month
movies_df["release_dow"] = movies_df["release_date"].dt.dayofweek  # Monday=0, Sunday=6

dow_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
dow_encoded = dow_encoder.fit_transform(movies_df[["release_dow"]])
dow_cols = [f"dow_{i}" for i in range(dow_encoded.shape[1])]
dow_encoded_df = pd.DataFrame(dow_encoded, columns=dow_cols)
movies_df = pd.concat([movies_df.reset_index(drop=True), dow_encoded_df.reset_index(drop=True)], axis=1)

def is_franchise(title):
    if isinstance(title, str) and (re.search(r"\d", title) or "Part" in title or "part" in title):
        return 1
    return 0

movies_df["is_franchise"] = movies_df["title"].apply(is_franchise)

np.random.seed(123)  ### for reproducibility, probably stupid
movies_df["has_star_power"] = np.random.randint(0, 2, size=len(movies_df)) # kill me

numeric_cols = [
    "budget",
    "runtime",
    "vote_average",
    "vote_count",
    "release_year",
    "release_month",
    "is_franchise",
    "has_star_power",
]

genre_cols = list(genre_encoded_df.columns)

# Day-of-week one-hot
dow_one_hot_cols = dow_cols

text_col = "overview"
all_features = numeric_cols + genre_cols + dow_one_hot_cols 

# target
y = movies_df["log_revenue"]

X_full = movies_df[all_features + [text_col]]
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector, make_column_transformer

# transformers
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

# text vectorizer
text_transformer = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=500 
    ))
])
passthrough_cols = genre_cols + dow_one_hot_cols

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("pass_genre_dow", "passthrough", passthrough_cols),
    ("text", text_transformer, text_col)
])

model_pipeline = Pipeline([
    ("preproc", preprocessor),
    ("gb", GradientBoostingRegressor(random_state=42))
])

param_distributions = {
    "gb__n_estimators": [50, 100, 200],
    "gb__max_depth": [3, 5, 7],
    "gb__learning_rate": [0.01, 0.1, 0.2],
    "gb__subsample": [0.8, 1.0],
}

search = RandomizedSearchCV(
    model_pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=1,
    random_state=42
)

search.fit(X_train, y_train)
print("\nBest hyperparameters:", search.best_params_)

best_pipeline = search.best_estimator_

y_pred = np.expm1(best_pipeline.predict(X_test))  # revert log transform
actual = np.expm1(y_test)

mse = mean_squared_error(actual, y_pred)
r2 = r2_score(actual, y_pred)
print(f"Test MSE: {mse:.5f}")
print(f"Test R^2: {r2:.5f}")

# Residual plot
residuals = y_test - best_pipeline.predict(X_test)  # in log space
plt.figure(figsize=(6, 5))
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residual Plot (Gradient Boosting)")
plt.xlabel("Actual Log-Revenue")
plt.ylabel("Residuals")
plt.show()

# Actual vs Predicted Revenue
plt.figure(figsize=(7, 7))
plt.scatter(actual, y_pred, alpha=0.5)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
         color='red', linestyle='--', label="Perfect Prediction")
plt.title("Actual vs Predicted Revenue (Gradient Boosting)")
plt.xlabel("Actual Revenue (billions)")
plt.ylabel("Predicted Revenue (billions)")
plt.legend()
plt.show()

# feature importances GradientBoostingRegressor
feature_names_num = numeric_cols
feature_names_pass = passthrough_cols

tfidf_vocab = search.best_estimator_["preproc"].named_transformers_["text"] \
              .named_steps["tfidf"].get_feature_names_out()

all_final_features = (
    feature_names_num
    + feature_names_pass
    + tfidf_vocab.tolist()
)

gb_model = best_pipeline["gb"]
importances = gb_model.feature_importances_

# sort by importance
fi_df = pd.DataFrame({
    "feature": all_final_features,
    "importance": importances
})
fi_df.sort_values("importance", ascending=False, inplace=True)

# top 20
top_n = 20
fi_top = fi_df.head(top_n).iloc[::-1]  # reverse
plt.figure(figsize=(6, 6))
plt.barh(fi_top["feature"], fi_top["importance"])
plt.title("Top 20 Feature Importances (Gradient Boosting)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

print(fi_top)
