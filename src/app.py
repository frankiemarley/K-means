import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
url = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
data = pd.read_csv(url)

# Define features and target
X = data[['Latitude', 'Longitude']]
y = data['MedInc']

# Print data summary
print("Data summary:")
print(X.describe())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Means clustering
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster assignments to the data DataFrame
data['Cluster'] = clusters

# Predict clusters for the test set
test_clusters = kmeans.predict(X_test)

# Plot the clusters including both train and test data
plt.figure(figsize=(12, 8))
scatter_train = plt.scatter(X_train['Longitude'], X_train['Latitude'], 
                            c=kmeans.predict(X_train), cmap='viridis', 
                            alpha=0.6, label='Train')
scatter_test = plt.scatter(X_test['Longitude'], X_test['Latitude'], 
                           c=test_clusters, cmap='viridis', 
                           marker='x', s=100, alpha=1, label='Test')
plt.title("K-Means Clustering of Housing Data (Train and Test)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(scatter_train, label='Cluster')
plt.legend()
plt.show()

# Income Distribution Visualization:
plt.figure(figsize=(12, 6))
data.boxplot(column='MedInc', by='Cluster')
plt.title('Median Income Distribution by Cluster')
plt.suptitle('')  # This removes the automatic suptitle
plt.show()

# Geographical Cluster Visualization:
plt.figure(figsize=(15, 10))
for cluster in range(6):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], 
                alpha=0.5, label=f'Cluster {cluster}')
plt.title('Geographical Distribution of Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()

# Cluster statistics
print("\nCluster statistics:")
print(data.groupby('Cluster').agg({
    'Latitude': ['mean', 'min', 'max'],
    'Longitude': ['mean', 'min', 'max'],
    'MedInc': ['mean', 'min', 'max']
}))

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, kmeans.predict(X_train))
y_pred = rf_model.predict(X_test)
y_test_clusters = kmeans.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test_clusters, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_clusters, y_pred))

# Elbow and Silhouette Score methods
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

# Dendrogram (using a sample of the data)
sample_size = 1000  # Adjust this value based on your computer's capability
sample_indices = np.random.choice(X.index, sample_size, replace=False)
X_sample = X.loc[sample_indices]

linkage_matrix = linkage(X_sample, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, truncate_mode='lastp', p=12, leaf_rotation=90, leaf_font_size=12, show_contracted=True)
plt.title('Dendrogram (Sampled Data)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
