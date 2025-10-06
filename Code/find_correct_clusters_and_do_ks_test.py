import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde, ks_2samp
from matplotlib.patches import Patch


def elbow_method(df):
    
    input_df = df[['CB', 'DCB', 'TOL', 'annealing_temperature', 'conductivity']].values
    scaler = StandardScaler()
    normalized_input_df = scaler.fit_transform(input_df)
    
    # Calculate inertia (Within-Cluster Sum of Squares) for different number of clusters
    inertia = []
    range_clusters = range(1, 21)  # Trying 1 to 20 clusters
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(normalized_input_df)
        inertia.append(kmeans.inertia_)
    
    
    plt.figure(figsize=(6, 6))
    plt.plot(range_clusters, inertia, marker='o', markersize=5, color='black')
    plt.xlabel('Number of Clusters', fontsize=24)
    plt.ylabel('WCSS', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xticks(np.arange(0, 21, 4))
    plt.show()
    return normalized_input_df


def generate_train_test_indices(num_clusters, normalized_input_df, random_state=42, samples_per_cluster=5):
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_input_df)
    test_indices = []
    for cluster in np.unique(cluster_labels):
        cluster_points = np.where(cluster_labels == cluster)[0]
        rng = np.random.RandomState(random_state)  # Set the random seed
        representative_indices = rng.choice(cluster_points, size=samples_per_cluster, replace=False)
        test_indices.extend(representative_indices)

    test_indices = np.array(test_indices)
    train_indices = np.setdiff1d(np.arange(len(normalized_input_df)), test_indices)

    return train_indices, test_indices


def plot_kde_train_test(train_indices, test_indices, df):

    columns = ['CB', 'DCB', 'TOL', 'annealing_temperature', 'conductivity']
    x_label_columns = ['% CB', '% DCB', '% TOL', 'Annealing Temperature (°C)', 'Conductivity(S/cm)']
    

    df_input = pd.DataFrame(df[['CB', 'DCB', 'TOL', 'annealing_temperature', 'conductivity']].values, 
                            columns=columns)


    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    
    for i, column in enumerate(columns):
        ax = axes[i]
        train_data = df_input[column].iloc[train_indices]
        test_data = df_input[column].iloc[test_indices]

        # Create the KDE estimators for train and test
        kde_train = gaussian_kde(train_data)
        kde_test = gaussian_kde(test_data)
        x_range = np.linspace(min(df_input[column]), max(df_input[column]), 1000)
        train_y = kde_train(x_range)
        test_y = kde_test(x_range)

        # Clip the y-values to remove negative densities
        train_y = np.clip(train_y, 0, None)
        test_y = np.clip(test_y, 0, None)

        ax.fill_between(x_range, train_y, color='blue', alpha=0.5)
        ax.fill_between(x_range, test_y, color='red', alpha=0.5)
        ax.set_xlabel(x_label_columns[i], fontsize=24)
        ax.set_ylabel('Density', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=24)
        
        if i == 3: 
            plt.xticks([30, 90,  150,  210,  270], fontsize=24)  # Custom ticks for 'annealing_temperature'
    # Sixth subplot for the shared legend
    axes[5].axis('off')
    legend_handles = [
        Patch(color='blue', alpha=0.5, label='Train'),
        Patch(color='red', alpha=0.5, label='Test')
    ]
    axes[5].legend(handles=legend_handles, loc='center', fontsize=24, frameon=False)

    plt.tight_layout()
    plt.show()

def perform_ks_test(train_indices, test_indices, df):
    ks_results = {}
    columns = ['CB', 'DCB', 'TOL', 'annealing_temperature', 'conductivity']
    
    for column in columns:
        train_data = df[column].iloc[train_indices]
        test_data = df[column].iloc[test_indices]
        ks_statistic, p_value = ks_2samp(train_data, test_data)
        ks_results[column] = {'KS Statistic': ks_statistic, 'p-value': p_value}
    return ks_results
