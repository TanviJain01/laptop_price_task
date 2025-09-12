import numpy as np
import pandas as pd
import pickle
import argparse
from train_model import RandomForest, DecisionTree, LinearRegressionGD, RidgeRegression, LassoRegression


def regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return mse, rmse, r2


def standardize_with_params(X, mu, sigma):
    return (X - mu) / sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--metrics_output_path", required=True)
    parser.add_argument("--predictions_output_path", required=True)
    args = parser.parse_args()

    # Load model
    with open(args.model_path, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    mu = model_data["mu"]
    sigma = model_data["sigma"]
    features = model_data["features"]
    target_col = model_data["target_col"]
    use_log = model_data["use_log"]

    # Load data
    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=[target_col]).copy()

    X = df[features].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col], _ = pd.factorize(X[col])

    X = X.values.astype(float)
    y_true_raw = df[target_col].values.reshape(-1, 1)
    y_true = np.log(y_true_raw) if use_log else y_true_raw

    # Standardize
    X_scaled = standardize_with_params(X, mu, sigma)

    # Predict
    y_pred = model.predict(X_scaled)

    # Metrics
    mse, rmse, r2 = regression_metrics(y_true, y_pred)

    with open(args.metrics_output_path, "w") as f:
        f.write(f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}\n")

    pd.DataFrame(y_pred).to_csv(args.predictions_output_path, index=False, header=False)

    print(f"Metrics saved to {args.metrics_output_path}")
    print(f"Predictions saved to {args.predictions_output_path}")
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")


if __name__ == "__main__":
    main()
