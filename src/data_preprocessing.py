#data exploration and processing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

data = pd.read_csv(r"C:\Users\tanvi\Documents\Tanvi_Jain_A1\laptop_price_task\data\Laptop Price.csv")

#remove characters from numerical data
data['Ram'] = data['Ram'].str.replace("GB", "")
data['Weight'] = data['Weight'].str.replace("kg", "")
data['Ram'] = data['Ram'].astype('int32')
data['Weight'] = data['Weight'].astype('float32')

#clean screenresolution - check touchscreen and ips indicators
data['Touchscreen'] = data['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
data['Ips'] = data['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
# get resolution data
data[['X_res','Y_res']] = data['ScreenResolution'].str.extract(r'(\d+)[xX](\d+)')
data['X_res'] = data['X_res'].astype(int)
data['Y_res'] = data['Y_res'].astype(int)

# PPI = diagonal resolution / inches
data['ppi'] = (((data['X_res']**2 + data['Y_res']**2) ** 0.5) / data['Inches']).astype(float)
data.drop(columns = ['ScreenResolution', 'Inches','X_res','Y_res'], inplace=True)

#clean memory
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

    for i, entry in df[col].items():
        if pd.isna(entry):
            continue

        parts = [p.strip() for p in entry.split("+")]

        for p in parts:
            
            size_match = re.search(r"(\d+\.?\d*)(TB|GB)", p)
            if not size_match:
                continue

            size_gb = size_to_gb(size_match.group(0))

            if "HDD" in p:
                df.at[i, "HDD_GB"] += size_gb
            elif "SSD" in p:
                df.at[i, "SSD_GB"] += size_gb
            elif "Flash" in p:
                df.at[i, "Flash_GB"] += size_gb
            elif "Hybrid" in p:
                df.at[i, "Hybrid_GB"] += size_gb

 
    data = data.drop(columns=[col])
    return data


#clean gpu
data['Gpu_brand'] = data['Gpu'].apply(lambda x: x.split()[0])
data = data[data['Gpu_brand'] != 'ARM'] #very little data for arm
data = data.drop(columns=['Gpu'])

#clean opsys
def cat_os(inp):
    if inp in ['Windows 10', 'Windows 7', 'Windows 10 S']:
        return 'Windows'
    elif inp in ['macOS', 'Mac OS X']:
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

data['os'] = data['OpSys'].apply(cat_os)
data = data.drop(columns=['OpSys'])

#check data
print(data.head())
print("\nData types:\n", data.dtypes)