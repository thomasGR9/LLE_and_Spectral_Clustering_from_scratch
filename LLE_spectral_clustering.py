import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def find_kneighbors_indexes(x, number_of_neighbors):
    NN = NearestNeighbors(n_neighbors=number_of_neighbors)
    NN.fit(x)
    return NN.kneighbors(return_distance=False)

class LLE:
    def __init__(self, n_components, n_neighbors):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

        self.eigenvectors = None

    def fit_transform(self, x):
        n_samples, n_dim = x.shape
        neighbor_indexes = find_kneighbors_indexes(x=x, number_of_neighbors=self.n_neighbors) # Find the indices of the k-nearest neighbors for each data point.
        weights = np.zeros(shape=(n_samples, n_samples))
        one_matrix = np.ones(shape=(1, self.n_neighbors))
        for i in range(n_samples):
            x_neigh_T = x[neighbor_indexes[i]].T # Get the transposed coordinates of the neighbors.
            comp = (x[i,:].reshape(n_dim,1)@one_matrix)-x_neigh_T
            Gi=comp.T @ comp
            trace = np.trace(Gi)
            if trace > 0:    
                Gi += np.eye(self.n_neighbors)*trace*0.0008
            else:                             # Add a small regularization term to ensure numerical stability.
                Gi += np.eye(self.n_neighbors) * 0.0008
            inverse_Gi = np.linalg.inv(Gi)
            weights[i,neighbor_indexes[i, :]]= one_matrix@inverse_Gi/(one_matrix@(one_matrix@inverse_Gi).T)[0][0]  # Compute and normalize the weights for this point.
        del x_neigh_T, comp, Gi, inverse_Gi
        print("Calculated weights")
        M = np.eye(n_samples)-weights
        M = M.T @ M    # Compute the embedding matrix
        del weights
        eigenvalues, eigenvectors = np.linalg.eig(M)
        del M
        print("Found eigenvectors")
        sorted_indeces = np.argsort(eigenvalues)    
        index = np.searchsorted(eigenvalues[sorted_indeces], 10**(-9), side='right') # Ignore near-zero eigenvalues and select the smallest non-zero eigenvalues.
        self.eigenvectors = eigenvectors[:, sorted_indeces[index:index+self.n_components]]
        return np.real(self.eigenvectors)


def rbf_matrix_fit(x, gamma):
    #This will split the matrix (of shape (x,x)) on 9 blocks and calculate the rbf kernel for the 6 lower half blocks with iterations.
    # On each iteration it will copy the transposed block on the mirrored one using the fact that the matrix is symmetric
    N = x.shape[0]
    if N <= 2000:
        block_size = max(1, N // 3) #ensure that is one when x<3. The 9 blocks i found ran faster on the iris dataset, so i picked it
    else:
        block_size = max(1, N // 6) #If N is large use smaller block size to reduce peak memory consumption
    result = np.zeros((N, N), dtype=np.float32)
    x = x.astype(np.float32)  # Convert inputs to float32
    gamma = np.float32(gamma)
    for i in range(0, N, block_size):
        for j in range(i, N, block_size):
            # Compute block indices
            x_block_i = x[i:i+block_size]
            x_block_j = x[j:j+block_size]
            # Compute pairwise differences for the current block
            diff = x_block_i[:, np.newaxis, :] - x_block_j[np.newaxis, :, :] #Vectorized calculations of differences on this block
            squared_diff = np.sum(diff**2, axis=2, dtype=np.float32)
            exp_block = np.exp(-gamma * squared_diff, dtype=np.float32)
            # Assign to the result matrix
            result[i:i+block_size, j:j+block_size] = exp_block
            if i != j:  # Mirror for symmetric positions
                result[j:j+block_size, i:i+block_size] = exp_block.T
    return result.astype(np.float64)

def compute_normalized_laplacian(affinity_matrix):
    n = affinity_matrix.shape[0]

    
    degree = np.sum(affinity_matrix, axis=1) # Compute the degree vector

    
    degree_inv_sqrt = 1.0 / np.sqrt(degree + 1e-10) # Compute D^(-1/2)


    
    normalized_laplacian = np.eye(n) - (affinity_matrix * degree_inv_sqrt[:, None]) * degree_inv_sqrt[None, :] # Avoid forming full diagonal matrices:

   
    np.fill_diagonal(normalized_laplacian, 0.5 * (normalized_laplacian.diagonal() + normalized_laplacian.T.diagonal()))  # Ensure symmetry by averaging with the transpose
    return normalized_laplacian


def eigengap_heuristic(sorted_eigenvalues, number_to_find, min_clusters):
    #Identifies the best candidates for the number of clusters based on the eigengap heuristic.
    eigenvalue_gaps = np.diff(sorted_eigenvalues[min_clusters-1:]) / sorted_eigenvalues[min_clusters-1:-1] # Compute the eigengaps

    
    largest_gaps_indices = np.argsort(eigenvalue_gaps)[::-1][:number_to_find] # Find the indices of the largest gaps

    
    proposed_clusters = [(index + min_clusters) for index in largest_gaps_indices] # Convert indices to cluster numbers and save as list

    return sorted(proposed_clusters)  #return sorted


def rbf_pred_func(x_train, x_test, gamma):
    M, N = x_train.shape
    m = x_test.shape[0]

    x_train = x_train.astype(np.float32)  # Convert inputs to float32
    x_test = x_test.astype(np.float32)
    gamma = np.float32(gamma)

    final_result = np.zeros((M, m), dtype=np.float32) 

    chunk_size = max(1, m // 3)
    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        x_test_chunk = x_test[start:end, :].astype(np.float32) 

        
        diff = x_train[:, np.newaxis, :] - x_test_chunk[np.newaxis, :, :]  # Shape (M, chunksize, N)
        squared_diff = np.sum(diff**2, axis=2, dtype=np.float32)  # Sum along the N dimension to get (M, chunksize)
        final_result[:, start:end] = np.exp(-gamma*squared_diff, dtype=np.float32)

    return final_result.astype(np.float64) #Convert back to float64 to avoid mixed presicion calculations

def cluster_preds_to_labels(cluster_preds, labels):
    counts = {num: {cls: 0 for cls in np.unique(labels)} for num in np.unique(cluster_preds)} # Create a nested dictionary to count occurrences of true labels in each cluster

    for num, cls in zip(cluster_preds, np.array(labels)):    # Populate the counts
        counts[num][cls] += 1  

    results_df = pd.DataFrame(counts) 
    results_norm_df = results_df.div(results_df.sum(axis=1), axis=0) # Normalize the DataFrame to calculate the proportion of each label in each cluster
    mapping_series = results_norm_df.idxmax() # Find the most likely label for each cluster based on the highest proportion
    return mapping_series[cluster_preds].values, mapping_series #Return the mapped predictions


    





class SpectralClustering:
    def __init__(self, n_clusters=2, gamma=1.0): 
        self.n_clusters = n_clusters
        self.gamma = gamma
        
        self.eigenvectors_ = None
        self.kmeans_ = None
        self.mapping_series = None
        

    def fit(self, X, y):

        affinity_matrix = rbf_matrix_fit(X, gamma=self.gamma)
        
        L = compute_normalized_laplacian(affinity_matrix)
        del affinity_matrix
        print("Calculated Normalized Laplacian Matrix")
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        del L
        sorted_indeces = np.argsort(eigenvalues)    
        self.eigenvectors_ = eigenvectors[:, sorted_indeces[:self.n_clusters]]
        
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        cluster_preds = self.kmeans_.fit_predict(self.eigenvectors_)
        _, self.mapping_series = cluster_preds_to_labels(cluster_preds=cluster_preds, labels=y)
        return self

    def predict(self, X_train, X_test, return_test_projections=False):
        affinity_preds = rbf_pred_func(x_train=X_train, x_test=X_test, gamma=self.gamma)
        projected_new = affinity_preds.T @ self.eigenvectors_
        cluster_preds = self.kmeans_.predict(projected_new)
        if return_test_projections==False:
            return self.mapping_series[cluster_preds].values
        elif return_test_projections==True:
            return self.mapping_series[cluster_preds].values, projected_new
        
        
        


    def fit_predict(self, X, y):

        affinity_matrix = rbf_matrix_fit(X, gamma=self.gamma)
        
        L = compute_normalized_laplacian(affinity_matrix)
        del affinity_matrix
        
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        del L
        sorted_indeces = np.argsort(eigenvalues)    
        self.eigenvectors_ = eigenvectors[:, sorted_indeces[:self.n_clusters]]
        
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        cluster_preds = self.kmeans_.fit_predict(self.eigenvectors_)
        preds, self.mapping_series = cluster_preds_to_labels(cluster_preds=cluster_preds, labels=y)
        return preds
    
    def find_best_number_of_clusters(self, X, number_to_find, min_clusters):
        affinity_matrix = rbf_matrix_fit(X, gamma=self.gamma)
        
        L = compute_normalized_laplacian(affinity_matrix)
        del affinity_matrix
        
        eigenvalues, _ = np.linalg.eigh(L)
        sorted_indeces = np.argsort(eigenvalues) 
        eigenvalues = eigenvalues[sorted_indeces]
        return eigengap_heuristic(sorted_eigenvalues=eigenvalues, number_to_find=number_to_find, min_clusters=min_clusters)
    
    def make_plot_1(self, x, y, eigen_projections, kmeans):
        if x.shape[1] != 2:
            raise ValueError("Only works for 2D data")
        import matplotlib.pyplot as plt
        cluster_predictions = kmeans.predict(eigen_projections)
        unique_labels = np.unique(y)
        markers = ['o', 's', 'D', '^', 'v', 'P', '*']  


        cluster_colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))

        plt.figure(figsize=(10, 7))


        for label, marker in zip(unique_labels, markers):
            label_indices = y == label
            for cluster_idx, color in enumerate(cluster_colors):
                cluster_indices = cluster_predictions == cluster_idx

                indices = label_indices & cluster_indices
                plt.scatter(
                    x[indices, 0], 
                    x[indices, 1], 
                    color=color, 
                    marker=marker, 
                    edgecolor='black', 
                    s=80, 
                )

        for label, marker in zip(unique_labels, markers):
            plt.scatter([], [], color='gray', marker=marker, label=f'Label {label}')

        
        for cluster_idx, color in enumerate(cluster_colors):
            plt.scatter([], [], color=color,marker='o', label=f'Cluster {cluster_idx}')


    
        plt.title('2D Visualization of Data with K-Means Predictions')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(loc='best', fontsize=9)
        plt.grid(True)
        plt.show()
        
        
    def make_plot_2(self, x, y, eigen_projections, kmeans, mapping):
        if x.shape[1] != 2:
            raise ValueError("Only works for 2D data")
        import matplotlib.pyplot as plt
        
        cluster_predictions = kmeans.predict(eigen_projections)
        

        correct_predictions = (mapping[cluster_predictions].values == y)

       
        unique_clusters = np.unique(cluster_predictions)
        markers = ['o', 's', 'D', '^', 'v', 'P', '*']  

        plt.figure(figsize=(10, 7))

        
        for cluster, marker in zip(unique_clusters, markers):
            cluster_indices = cluster_predictions == cluster
            plt.scatter(
                x[cluster_indices & correct_predictions, 0], 
                x[cluster_indices & correct_predictions, 1], 
                color='green', 
                marker=marker, 
                edgecolor='black', 
                s=80, 
                label=f'Cluster {cluster} (Correct)'
            )
            plt.scatter(
                x[cluster_indices & ~correct_predictions, 0], 
                x[cluster_indices & ~correct_predictions, 1], 
                color='red', 
                marker=marker, 
                edgecolor='black', 
                s=80, 
                label=f'Cluster {cluster} (Incorrect)'
            )



        plt.title('2D Visualization of Data with Cluster Prediction Accuracy')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(loc='best', fontsize=9)
        plt.grid(True)
        plt.show()
    
