import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Reducing to 2 dimensions for visualization
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with PCA components
data_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
data_pca['MedInc'] = y

# Print explained variance ratio
print(f'Explained variance ratio of PCA components: {pca.explained_variance_ratio_}')

# Determine the optimal number of clusters using the Elbow Method
wcss = []
k_range = range(1, 11)  # Test from 1 to 10 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(12, 6))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means with the chosen number of clusters
optimal_k = 6  # Set to the optimal number of clusters from the Elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Add cluster assignments to the PCA DataFrame
data_pca['Cluster'] = clusters

# Plot the PCA components with cluster assignments
plt.figure(figsize=(10, 7))
scatter = plt.scatter(data_pca['PCA1'], data_pca['PCA2'], c=data_pca['Cluster'], cmap='viridis', alpha=0.6)
plt.title("PCA of K-Means Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label='Cluster')
plt.show()

# Predict clusters for the test set
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)
test_clusters_pca = kmeans.predict(X_test_pca)

# Add the test points to the PCA plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(data_pca['PCA1'], data_pca['PCA2'], c=data_pca['Cluster'], cmap='viridis', alpha=0.6, label='Train')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_clusters_pca, cmap='viridis', marker='x', s=100, alpha=1, label='Test')
plt.title("PCA of K-Means Clustering with Test Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.show()

# Train a Random Forest Classifier on PCA-transformed features
rf_model_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_pca.fit(X_pca, clusters)
y_pred_pca = rf_model_pca.predict(X_test_pca)
y_test_clusters_pca = kmeans.predict(X_test_pca)

print("\nClassification Report (PCA features):")
print(classification_report(y_test_clusters_pca, y_pred_pca))
print("\nConfusion Matrix (PCA features):")
print(confusion_matrix(y_test_clusters_pca, y_pred_pca))

# Correlation Analysis
correlation = data[['Latitude', 'Longitude', 'MedInc']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
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
