import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# loading cvs
movies_df = pd.read_csv("sample_data/tmdb_5000_movies.csv")

movies_df = movies_df[(movies_df["revenue"] > 0) & (movies_df["revenue"] <= 100_000_000)]
movies_df["revenue"] = movies_df["revenue"] / 1_000_000_000
movies_df["budget"] = movies_df["budget"] / 1_000_000_000
movies_df["log_revenue"] = np.log1p(movies_df["revenue"])


movies_df.dropna(subset=["budget", "revenue", "genres", "runtime"], inplace=True)

def extract_main_genre(genre_json):
    try:
        genres = json.loads(genre_json.replace("'", "\""))  # Replace single quotes with double quotes
        return genres[0]["name"] if genres else None
    except (json.JSONDecodeError, TypeError):
        return None

movies_df["genre_main"] = movies_df["genres"].apply(extract_main_genre)

movies_df.dropna(subset=["genre_main"], inplace=True)

# One-hot   main genre
encoder = OneHotEncoder(sparse_output=False)
genre_encoded = encoder.fit_transform(movies_df[["genre_main"]])
genre_encoded_df = pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out(["genre_main"]))

# merge with original dataframe
movies_df = pd.concat([movies_df.reset_index(drop=True), genre_encoded_df.reset_index(drop=True)], axis=1)

# features = ["budget", "runtime", "vote_average"] + list(genre_encoded_df.columns)

numeric_features = ["budget", "runtime", "vote_average","vote_count"]

# pipeline for polynomial expansion + linear regression
poly_degree = 3  # Try 2 or 3 (be cautious with 3+ as it can quickly add many features)
poly_model = Pipeline([
    ("scaler", StandardScaler()),                    # Scale numeric data
    ("poly", PolynomialFeatures(degree=poly_degree,
                                include_bias=False)),  # Generate polynomial features
    ("lin_reg", LinearRegression())
])

# target = "revenue"

X = movies_df[numeric_features + list(genre_encoded_df.columns)]
y = movies_df["log_revenue"]

# scale numeric
scaler = StandardScaler()
X[["budget", "runtime", "vote_average","vote_count"]] = scaler.fit_transform(X[["budget", "runtime", "vote_average","vote_count"]])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# mdel Initialization
lin_reg = LinearRegression()

# Cross-validation
cv_scores = cross_val_score(lin_reg, X_train, y_train, cv=5, scoring='r2')
print("Cross-Validation R^2 scores:", cv_scores)
print("Mean CV R^2 score:", np.mean(cv_scores))

# train the model
lin_reg.fit(X_train, y_train)

y_pred = np.expm1(lin_reg.predict(X_test))

predicted_revenue = y_pred
actual_revenue = np.expm1(y_test)

# MSE and R^2
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test MSE:", mse)
print("Test R^2:", r2)

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Actual Revenue")
plt.ylabel("Residuals")
plt.show()

# Predicted vs Actual
plt.figure(figsize=(8, 8))
plt.scatter(actual_revenue, predicted_revenue, alpha=0.5)
plt.plot([actual_revenue.min(), actual_revenue.max()],
         [actual_revenue.min(), actual_revenue.max()],
         color='red', linestyle='--', label="Perfect Prediction")

plt.xlim([actual_revenue.min(), actual_revenue.max()])
plt.ylim([actual_revenue.min(), actual_revenue.max()])

# todo add labels and title
plt.title("Actual vs Predicted Revenue")
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.legend()
plt.grid(True)
plt.show()

plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

coefficients = lin_reg.coef_
feature_names = X.columns

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

top_features = feature_importance.reindex(feature_importance["Coefficient"].abs().sort_values(ascending=False).index)

top_n = 20
top_features = top_features.head(top_n)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Coefficient"])
plt.title("Top 20 Feature Importance")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()

# Cross-validation plot
plt.bar(range(1, len(cv_scores) + 1), cv_scores)
plt.title("Cross-Validation R^2 Scores")
plt.xlabel("Fold")
plt.ylabel("R^2 Score")
plt.show()

metrics = pd.DataFrame({
    "Metric": ["MSE", "R^2"],
    "Value": [mse, r2]
})

print(metrics)

# Compare revenue
sns.kdeplot(y_test, label="Actual Revenue", shade=True)
sns.kdeplot(y_pred, label="Predicted Revenue", shade=True)
plt.title("Revenue Distribution Comparison")
plt.xlabel("Revenue")
plt.ylabel("Density")
plt.legend()
plt.show()
