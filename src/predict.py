import numpy as np
import pandas as pd
import pickle
import argparse
import re

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

def standardize_with_params(X, mu, sigma):
    return (X - mu) / sigma

def parse_memory(data, col="Memory"):
    data["HDD_GB"] = 0
    data["SSD_GB"] = 0
    data["Flash_GB"] = 0
    data["Hybrid_GB"] = 0

    def size_to_gb(size_str):
        if "TB" in size_str:
            return float(size_str.replace("TB", "").strip()) * 1000
        elif "GB" in size_str:
            return float(size_str.replace("GB", "").strip())
        return 0

    for i, entry in data[col].items():
        if pd.isna(entry):
            continue

        parts = [p.strip() for p in entry.split("+")]

        for p in parts:
            size_match = re.search(r"(\d+\.?\d*)(TB|GB)", p)
            if not size_match:
                continue

            size_gb = size_to_gb(size_match.group(0))

            if "HDD" in p:
                data.at[i, "HDD_GB"] += size_gb
            elif "SSD" in p:
                data.at[i, "SSD_GB"] += size_gb
            elif "Flash" in p:
                data.at[i, "Flash_GB"] += size_gb
            elif "Hybrid" in p:
                data.at[i, "Hybrid_GB"] += size_gb

    data = data.drop(columns=[col])
    return data

def cat_os(inp):
    if inp in ['Windows 10', 'Windows 7', 'Windows 10 S']:
        return 'Windows'
    elif inp in ['macOS', 'Mac OS X']:
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--metrics_output_path', required=True)
    parser.add_argument('--predictions_output_path', required=True)
    args = parser.parse_args()
    
    with open(args.model_path, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    mu = model_data['mu']
    sigma = model_data['sigma']
    features = model_data['features']
    target_col = model_data['target_col']
    
    data = pd.read_csv(args.data_path)
    
    data['Ram'] = data['Ram'].str.replace("GB", "")
    data['Weight'] = data['Weight'].str.replace("kg", "")
    data['Ram'] = data['Ram'].astype('int32')
    data['Weight'] = data['Weight'].astype('float32')

    data['Touchscreen'] = data['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
    data['Ips'] = data['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
    data[['X_res','Y_res']] = data['ScreenResolution'].str.extract(r'(\d+)[xX](\d+)')
    data['X_res'] = data['X_res'].astype(int)
    data['Y_res'] = data['Y_res'].astype(int)

    data['ppi'] = (((data['X_res']**2 + data['Y_res']**2) ** 0.5) / data['Inches']).astype(float)
    data.drop(columns = ['ScreenResolution', 'Inches','X_res','Y_res'], inplace=True)

    data = parse_memory(data)

    data['Gpu_brand'] = data['Gpu'].apply(lambda x: x.split()[0])
    data = data[data['Gpu_brand'] != 'ARM']
    data = data.drop(columns=['Gpu'])

    data['os'] = data['OpSys'].apply(cat_os)
    data = data.drop(columns=['OpSys'])
    
    data = data.dropna(subset=[target_col]).copy()
    
    X = data.drop(columns=[target_col])
    y_true = data[target_col].values.reshape(-1, 1)
    
    for col in X.select_dtypes(include="object").columns:
        X[col], _ = pd.factorize(X[col])
    
    X = X.values.astype(float)
    Xs = standardize_with_params(X, mu, sigma)
    
    y_pred = model.predict(Xs)
    
    mse_val = mse(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    
    with open(args.metrics_output_path, "w") as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse_val:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse_val:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2_val:.2f}\n")
    
    pred_df = pd.DataFrame(y_pred.ravel())
    pred_df.to_csv(args.predictions_output_path, index=False, header=False)
    
    print(f"Metrics saved to: {args.metrics_output_path}")
    print(f"Predictions saved to: {args.predictions_output_path}")
    print(f"MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, R²: {r2_val:.2f}")

if __name__ == "__main__":
    main()