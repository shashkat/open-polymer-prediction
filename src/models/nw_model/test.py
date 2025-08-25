# just testing some functions from the nadaraya-watson model.

import pandas as pd
import numpy as np

df_train = pd.read_csv('data/raw/neurips-open-polymer-prediction-2025/train.csv')
data_tg = df_train.loc[~df_train['Tg'].isna(), ['id', 'SMILES', 'Tg']]

bandwidth = 15
embeddings = np.random.randn(7973,1024)
test_embeddings = np.random.randn(100, 1024)

def kernel_fn(x):
    numerator = -(x * x)
    denominator = 2 * bandwidth * bandwidth
    return np.exp(numerator/denominator)

def distance(arr1, arr2):
    # Compute pairwise distances (Euclidean by default)
    arr1_exp = arr1[:, np.newaxis, :]  # (x, 1, 120)
    arr2_exp = arr2[np.newaxis, :, :]  # (1, y, 120)
    sq_diff = (arr1_exp - arr2_exp) ** 2  # (x, y, 120)
    sq_distances = np.sum(sq_diff, axis=2)  # (x, y)
    distances = np.sqrt(sq_distances)  # (x, y)
    return distances

def predict_tg(test_embeddings):
    # find out the distances between test_embeddings and self.embeddings' subset of self.data_tg
    dists = distance(embeddings[data_tg.index], test_embeddings) # shape: nrow(data_tg),test_embeddings
    weights = kernel_fn(dists) # pass the mere distances to kernel to get weights
    # now, we scale the kernel-passed weights so that for each test_embedding, the weights' sum is 1
    weights_scaled = weights/weights.sum(axis = 0)
    # now we just tranpose as test_embeddings being as rownames are more intuitive
    weights_scaled = weights_scaled.T # shape: test_embeddings,nrow(data_tg)
    # now, just perform dot product with tg values vector to get tg value for each test_embedding
    train_tg_values = data_tg.iloc[:, -1].values
    output_property_values = np.matmul(weights_scaled, train_tg_values)
    return output_property_values # shape: (test_embeddings,)

temp = np.empty(0)
np.concatenate([temp,output_property_values]).shape



