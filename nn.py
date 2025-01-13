import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    movies_df = pd.read_csv(filepath)

    movies_df = movies_df[(movies_df["revenue"] > 0) & (movies_df["revenue"] <= 100_000_000)]
    movies_df["revenue"] = movies_df["revenue"] / 1_000_000_000
    movies_df["budget"] = movies_df["budget"] / 1_000_000_000

    movies_df.dropna(subset=["budget", "revenue", "genres", "runtime"], inplace=True)

    return movies_df


def extract_main_genre(genre_json: str) -> str:
    try:
        # Convert single quotes to double quotes to parse
        genres = json.loads(genre_json.replace("'", "\""))
        return genres[0]["name"] if genres else None
    except (json.JSONDecodeError, TypeError, IndexError):
        return None


def encode_genre(df: pd.DataFrame) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=False)
    genre_encoded = encoder.fit_transform(df[["genre_main"]])
    genre_encoded_df = pd.DataFrame(
        genre_encoded,
        columns=encoder.get_feature_names_out(["genre_main"])
    )
    # Merge with original
    df = pd.concat([df.reset_index(drop=True), genre_encoded_df], axis=1)
    return df


def get_polynomial_features(df: pd.DataFrame,
                           numeric_features: list,
                           degree: int = 2) -> pd.DataFrame:

    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_numeric_poly = poly.fit_transform(df[numeric_features])
    poly_feature_names = poly.get_feature_names_out(numeric_features)
    X_numeric_poly_df = pd.DataFrame(X_numeric_poly, columns=poly_feature_names)
    return X_numeric_poly_df


def remove_outliers_zscore(df: pd.DataFrame,
                           columns: list,
                           zscore_threshold: float = 3.0) -> pd.DataFrame:
    """
     z-score threshold (in absolute value).
    """
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std(ddof=0)  # population std if ddof=0, sample if ddof=1
        # Keep rows within the threshold
        df = df[np.abs((df[col] - mean_val) / (std_val + 1e-12)) < zscore_threshold]
    return df


def main():
    movies_df = load_and_preprocess_data("sample_data/tmdb_5000_movies.csv")
    movies_df["genre_main"] = movies_df["genres"].apply(extract_main_genre)
    movies_df.dropna(subset=["genre_main"], inplace=True)

    # One-hot
    movies_df = encode_genre(movies_df)

    numeric_cols_for_outliers = ["budget", "runtime", "vote_average", "vote_count", "revenue"]

    movies_df = remove_outliers_zscore(movies_df, numeric_cols_for_outliers, zscore_threshold=2.2)

    numeric_features = ["budget", "runtime", "vote_average", "vote_count"]

    X_numeric_poly_df = get_polynomial_features(movies_df, numeric_features, degree=2)

    genre_columns = [col for col in movies_df.columns if col.startswith("genre_main_")]
    X = pd.concat([X_numeric_poly_df, movies_df[genre_columns].reset_index(drop=True)], axis=1)

    # Target: revenue
    y = movies_df["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )
    nn_model.fit(X_train_scaled, y_train)

    y_pred_train = nn_model.predict(X_train_scaled)
    y_pred_test = nn_model.predict(X_test_scaled)

    # convert back: y_pred_test_exp = np.expm1(y_pred_test)
    y_pred_train_exp = y_pred_train
    y_pred_test_exp = y_pred_test
    y_train_exp = y_train
    y_test_exp = y_test

    residuals = y_test_exp - y_pred_test_exp
    residuals = np.array(residuals)
    # residuals = pd.Series(y_test_exp - y_pred_test_exp, index=np.arange(len(y_test_exp)))
    outlier_threshold = 0.05  # Adjust this threshold as needed
    outlier_indices = np.where(abs(residuals) > outlier_threshold)[0]

    residuals = np.delete(residuals, outlier_indices)
    X_test = np.delete(X_test, outlier_indices, axis=0)
    y_test_exp = np.delete(y_test_exp, outlier_indices, axis=0)
    y_pred_test_exp = np.delete(y_pred_test_exp, outlier_indices, axis=0)


    train_mse = mean_squared_error(y_train_exp, y_pred_train_exp)
    test_mse = mean_squared_error(y_test_exp, y_pred_test_exp)
    train_r2 = r2_score(y_train_exp, y_pred_train_exp)
    test_r2 = r2_score(y_test_exp, y_pred_test_exp)

    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)
    print("Train R^2:", train_r2)
    print("Test R^2:", test_r2)

    # Predicted vs. Actual
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_exp, y_pred_test_exp, alpha=0.5, label="Predictions")

    #red line for ideal prediction
    min_val, max_val = min(y_test_exp.min(), y_pred_test_exp.min()), max(y_test_exp.max(), y_pred_test_exp.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")
    plt.title("Actual vs. Predicted Revenue")
    plt.xlabel("Actual Revenue")
    plt.ylabel("Predicted Revenue")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Residual Plot
    # residuals = y_test_exp - y_pred_test_exp
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_exp, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Actual Revenue")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.show()

    # Revenue distr
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_test_exp, label="Actual Revenue", shade=True)
    sns.kdeplot(y_pred_test_exp, label="Predicted Revenue", shade=True)
    plt.title("Revenue Distribution Comparison")
    plt.xlabel("Revenue")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
