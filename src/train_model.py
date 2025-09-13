import numpy as np
import pickle
import os
import pandas as pd
from data_preprocessing import preprocess_data

def mse(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    return float(np.mean((y_true - y_pred)**2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res/ss_tot)

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
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n = X_b.shape[1]
        I = np.eye(n)
        I[0, 0] = 0
        self.theta = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ (X_b.T @ y)
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

class LassoRegression:
    def __init__(self, alpha=0.1, lr=0.001, epochs=5000):
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape
        self.theta = np.zeros((n, 1))
        for _ in range(self.epochs):
            y_pred = X_b @ self.theta
            grad = (2/m) * X_b.T @ (y_pred - y) + self.alpha * np.sign(self.theta)
            grad[0] -= self.alpha * np.sign(self.theta[0])
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
            split_idx, split_thr, mse_val = self._best_split(X, y, feat_idx)
            if split_idx is not None and mse_val < best["mse"]:
                best = {"mse": mse_val, "feature": split_idx, "threshold": split_thr}
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
        return np.array([self._predict_sample(x, self.tree) for x in X]).reshape(-1, 1)

def train_models(data_path):
    X, y, mu, sigma, features, target_col = preprocess_data(data_path)
    
    models = [
        ("regression_model1.pkl", LinearRegressionGD(lr=0.01, epochs=5000)),
        ("regression_model2.pkl", RidgeRegression(alpha=10.0)),
        ("regression_model3.pkl", LassoRegression(alpha=0.1, lr=0.001, epochs=5000)),
        ("regression_model_final.pkl", DecisionTree(max_depth=12, min_samples_split=5))
    ]
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    for model_name, model in models:
        model.fit(X, y)
        y_pred = model.predict(X)
        
        model_data = {
            'model': model,
            'mu': mu,
            'sigma': sigma,
            'features': features,
            'target_col': target_col
        }
        
        with open(f"models/{model_name}", "wb") as f:
            pickle.dump(model_data, f)
        
        if model_name == "regression_model_final.pkl":
            mse_val = mse(y, y_pred)
            rmse_val = rmse(y, y_pred)
            r2_val = r2_score(y, y_pred)
            
            with open("results/train_metrics.txt", "w") as f:
                f.write("Regression Metrics:\n")
                f.write(f"Mean Squared Error (MSE): {mse_val:.2f}\n")
                f.write(f"Root Mean Squared Error (RMSE): {rmse_val:.2f}\n")
                f.write(f"R-squared (R²) Score: {r2_val:.2f}\n")
            
            pred_df = pd.DataFrame(y_pred.ravel())
            pred_df.to_csv("results/train_predictions.csv", index=False, header=False)

if __name__ == "__main__":
    train_models("data/Laptop Price.csv")