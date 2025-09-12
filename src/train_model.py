import numpy as np
import pandas as pd
import pickle
import os
from data_preprocessing import data   # assumes `data` is your preprocessed DataFrame


# ---------- Utilities ----------
def factorize_dataframe(df):
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object").columns:
        df_enc[col], _ = pd.factorize(df_enc[col])
    return df_enc

def standardize_features(df):
    mu = df.mean(axis=0)
    sigma = df.std(axis=0).replace(0, 1)
    X_scaled = (df - mu) / sigma
    return X_scaled.values.astype(float), mu.values, sigma.values

def regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return mse, rmse, r2


# ---------- Models ----------
class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=5000):
        self.lr = lr
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.zeros((X_b.shape[1], 1))
        m = X_b.shape[0]
        for _ in range(self.epochs):
            y_pred = X_b @ self.theta
            grad = (2/m) * X_b.T @ (y_pred - y)
            self.theta -= self.lr * grad
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta


class RidgeRegression:
    def __init__(self, lam=1.0):
        self.lam = lam
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n = X_b.shape[1]
        I = np.eye(n)
        I[0, 0] = 0  # no penalty for intercept
        self.theta = np.linalg.pinv(X_b.T @ X_b + self.lam * I) @ (X_b.T @ y)
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta


class LassoRegression:
    def __init__(self, lam=0.1, lr=0.001, epochs=5000):
        self.lam = lam
        self.lr = lr
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape
        self.theta = np.zeros((n, 1))
        for _ in range(self.epochs):
            y_pred = X_b @ self.theta
            grad = (2/m) * X_b.T @ (y_pred - y) + self.lam * np.sign(self.theta)
            grad[0] -= self.lam * np.sign(self.theta[0])  # no penalty on intercept
            self.theta -= self.lr * grad
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else self.n_features
        self.tree = self._grow_tree(X, y)

    def _mse(self, y):
        return np.var(y) * len(y)

    def _best_split(self, X, y, feat_idx):
        best_mse, split_idx, split_thr = float("inf"), None, None
        thresholds = np.unique(X[:, feat_idx])
        for t in thresholds:
            left_mask = X[:, feat_idx] <= t
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            mse = self._mse(y[left_mask]) + self._mse(y[right_mask])
            if mse < best_mse:
                best_mse, split_idx, split_thr = mse, feat_idx, t
        return split_idx, split_thr, best_mse

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        if (depth >= self.max_depth) or (n_samples < self.min_samples_split):
            return {"value": y.mean()}
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best = {"mse": float("inf")}
        for feat_idx in feat_idxs:
            split_idx, split_thr, mse = self._best_split(X, y, feat_idx)
            if split_idx is not None and mse < best["mse"]:
                best = {"mse": mse, "feature": split_idx, "threshold": split_thr}
        if best["mse"] == float("inf"):
            return {"value": y.mean()}
        left_mask = X[:, best["feature"]] <= best["threshold"]
        right_mask = ~left_mask
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth+1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth+1)
        return {"feature": best["feature"], "threshold": best["threshold"], "left": left_child, "right": right_child}

    def _predict_sample(self, inputs, tree):
        if "value" in tree:
            return tree["value"]
        if inputs[tree["feature"]] <= tree["threshold"]:
            return self._predict_sample(inputs, tree["left"])
        else:
            return self._predict_sample(inputs, tree["right"])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForest:
    def __init__(self, n_estimators=30, max_depth=12, min_samples_split=5, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.max_features or int(np.sqrt(X.shape[1]))
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds, axis=0).reshape(-1, 1)


# ---------- Training ----------
def train_models():
    X = data.drop(columns=["Price"])
    y_raw = data["Price"].values.reshape(-1, 1)
    y_log = np.log(y_raw)

    # Preprocessing
    X_enc = factorize_dataframe(X)
    X_scaled, mu, sigma = standardize_features(X_enc)

    models = [
        ("regression_model1.pkl", LinearRegressionGD(lr=0.01, epochs=5000), y_raw),
        ("regression_model2.pkl", RidgeRegression(lam=10), y_raw),
        ("regression_model3.pkl", LassoRegression(lam=0.1, lr=0.001, epochs=5000), y_raw),
        ("regression_model_final.pkl", RandomForest(n_estimators=30, max_depth=12), y_log),  # log target
    ]

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for name, model, target in models:
        model.fit(X_scaled, target)
        y_pred = model.predict(X_scaled)
        mse, rmse, r2 = regression_metrics(target, y_pred)

        print(f"{name[:-4]} -> MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

        # Save model
        model_data = {
            "model": model,
            "mu": mu,
            "sigma": sigma,
            "features": X.columns.tolist(),
            "target_col": "Price",
            "use_log": (name == "regression_model_final.pkl")
        }
        with open(f"models/{name}", "wb") as f:
            pickle.dump(model_data, f)

        # Save metrics/predictions for final model
        if name == "regression_model_final.pkl":
            with open("results/train_metrics.txt", "w") as f:
                f.write(f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}\n")
            pd.DataFrame(y_pred).to_csv("results/train_predictions.csv", index=False, header=False)


if __name__ == "__main__":
    train_models()
