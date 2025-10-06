import pandas as pd
import numpy as np
from scipy.integrate import simps
from scipy.signal import savgol_filter
import os
import re
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



def get_all_csv_files_path(data_folder, measurement_type, train_indices, test_indices):
    all_files_train = []
    all_files_test = []
    for foldername in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, foldername)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.csv') and measurement_type in filename:
                    match = re.search(r'Sample(\d+)', filename)  
                    experiment_number = match.group(1) if match else None 
                    index = int(experiment_number) - 1
                    if index in test_indices:
                        file_path = os.path.join(folder_path, filename)
                        all_files_test.append((file_path, foldername))
                        
                    else:
                        file_path = os.path.join(folder_path, filename)
                        all_files_train.append((file_path, foldername))
    return all_files_train, all_files_test


# Calculate Statistics for model evaluation

def calculate_statistics(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = y_pred_train.reshape(-1, 1)
    y_pred_test = y_pred_test.reshape(-1, 1)
    
    # Calculate R^2, RMSE, MAE, Kendall Tau, Pearson Correlation
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    kendall_tau_train = kendalltau(y_train.flatten(), y_pred_train.flatten()).correlation
    kendall_tau_test = kendalltau(y_test.flatten(), y_pred_test.flatten()).correlation

    pearson_corr_train, _ = pearsonr(y_train.flatten(), y_pred_train.flatten())
    pearson_corr_test, _ = pearsonr(y_test.flatten(), y_pred_test.flatten())

    print(f"Train R^2: {r2_train:.4f}, Test R^2: {r2_test:.4f}")
    print(f"Train RMSE: {rmse_train:.2f}, Test RMSE: {rmse_test:.2f}")
    print(f"Train MAE: {mae_train:.2f}, Test MAE: {mae_test:.2f}")
    print(f"Kendall Tau Train: {kendall_tau_train:.4f}, Test: {kendall_tau_test:.4f}")
    print(f"Pearson Correlation Train: {pearson_corr_train:.4f}, Test: {pearson_corr_test:.4f}")
    return y_pred_train, y_pred_test, r2_train, r2_test, rmse_train, rmse_test, mae_train, mae_test, kendall_tau_train, kendall_tau_test, pearson_corr_train, pearson_corr_test


# Plot Predicted vs Actual
def plotting(y_train, y_test, y_pred_train, y_pred_test):
    plt.figure(figsize=(4, 4))
    plt.scatter(y_train, y_pred_train, color='black', alpha=0.7, label='Train Data')
    plt.scatter(y_test, y_pred_test, color='red', alpha=0.7, label='Test Data')
    plt.plot([min(np.concatenate((y_train, y_test))), max(np.concatenate((y_train, y_test)))],
         [min(np.concatenate((y_train, y_test))), max(np.concatenate((y_train, y_test)))],
         color='blue', linestyle='--')
    plt.xlabel('True Conductivity')
    plt.ylabel('Predicted Conductivity')
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_all_feature_product(df):
    df_products = pd.DataFrame()
    df_products['exp_id'] = df['exp_id']
    feature_columns = [col for col in df.columns if col !='exp_id']

    for i in range(len(feature_columns)):
        for j in range(i + 1, len(feature_columns)):
            product_name = f"{feature_columns[i]}*{feature_columns[j]}"
            df_products[product_name] = df[feature_columns[i]] * df[feature_columns[j]]
    
    
    return df_products