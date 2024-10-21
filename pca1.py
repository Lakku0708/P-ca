import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target
df = pd.DataFrame(data, columns=iris.feature_names)
df['species'] = target

# Quick EDA using seaborn pairplot
sns.pairplot(df, hue="species", palette="Set2")
plt.show()

# Standardize the features
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Compute the covariance matrix
cov_matrix = np.cov(data_standardized.T)
print("Covariance Matrix:\n", cov_matrix)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Sort eigenvalues and eigenvectors
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project the data onto the first two principal components
pca_data = np.dot(data_standardized, eigenvectors[:, :2])

# Create a DataFrame for the PCA data with the target
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['species'] = target

# Plot the PCA result
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, palette="Set2", s=100)
plt.title("PCA of Iris Dataset")
plt.show()
