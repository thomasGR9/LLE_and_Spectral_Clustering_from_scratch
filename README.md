# **LLE & Spectral Clustering from Scratch**

This repository provides **from-scratch implementations** of two popular machine learning techniques:

- **Local Linear Embedding (LLE)**
- **Spectral Clustering**

Both implementations are written in Python

---

## **Features**

### **Local Linear Embedding (LLE)**
- **Neighborhood Search**:  
  - Uses `NearestNeighbors` from scikit-learn to efficiently find the k-nearest neighbors.
- **Reconstruction Weights**:  
  - Computes reconstruction weights for each point in the dataset with added regularization for numerical stability.
- **Efficient Eigen-Decomposition**:  
  - Solves the eigenvalue problem to extract the embedding vectors.
- **Memory Efficiency**:  
  - Minimizes memory usage by deleting intermediate variables when they are no longer needed.

### **Spectral Clustering**
- **Affinity Matrix Calculation**:  
  - Computes the RBF (Gaussian) kernel in a block-wise fashion to reduce peak memory consumption.
- **Normalized Laplacian**:  
  - Constructs a normalized Laplacian matrix in a memory-efficient manner.
- **Eigen-Decomposition**:  
  - Uses eigenvalue decomposition to extract clustering-relevant features.
- **Cluster Assignment**:  
  - Implements K-Means on the eigenvector projections.
- **Cluster Label Mapping**:  
  - Maps cluster predictions back to the original labels for evaluation.
- **Visualization Tools**:  
  - Provides functions for 2D visualization of clustering results and prediction accuracy.

