# HERE IS THE MAIN CODE USED BY SHASHANK FOR THE NADARAYA-WATSON METHOD ATTEMPT (THE ONE IN WHICH 
# THE PROPERTIES WERE OBTAINED BY DOING THE WEIGHTED AVERAGE OF THE PROPERTIES FROM TRAINING DATA). 
# WEIGHT BETWEEN ANY PAIR OF SMILES WAS OBTAINED BY COMPUTING EUCLIDEAN DISTANCE BETWEEN THEIR SENTENCE 
# TRANSFORMER EMBEDDINGS AND TRANSFORMING IT USING A GAUSSIAN KERNEL. SO FOR THIS SETUP, THE 
# HYPERPARAMETERS WERE: WHICH SENTENCE-TRANSFORMER MODEL TO USE, AND THE KERNEL (WHAT BANDWIDTH IF 
# THE GAUSSIAN KERNEL). 
# IN THE SUBMISSION ON KAGGLE, I ENDED UP USING A PUBLIC CALLED MODEL ChemBERTa-77M-MLM (https://www.kaggle.com/models/michaelrowen/c/Transformers/default)

# imports
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

# mse between all the values we have in df_train_subset2, vs what we predicted
def MSE(results, df_train_subset2):
    # now, for all the values we did have in df_train_subset2, we will see how different our predictions were from them
    mask = df_train_subset2.iloc[:, -5:].isna().values
    # convert results to array
    results_arr = pd.DataFrame(results).values
    diff = results_arr - df_train_subset2.iloc[:, -5:].values
    mse = (diff[~mask]*diff[~mask]).mean()
    return mse

# model = SentenceTransformer("Derify/ChemMRL-alpha")
model = SentenceTransformer('/Users/shashankkatiyar/Documents/learning_ml/open_polymer_prediction/models/ChemBERTa-77M-MLM')

# read data
df_train = pd.read_csv('data/raw/neurips-open-polymer-prediction-2025/train.csv')
df_test = pd.read_csv('data/raw/neurips-open-polymer-prediction-2025/test.csv')

# get all the subsets of the df, where each subset has all the information for one property
data_tg = df_train.loc[~df_train['Tg'].isna(), ['id', 'SMILES', 'Tg']]
data_ffv = df_train.loc[~df_train['FFV'].isna(), ['id', 'SMILES', 'FFV']]
data_tc = df_train.loc[~df_train['Tc'].isna(), ['id', 'SMILES', 'Tc']]
data_density = df_train.loc[~df_train['Density'].isna(), ['id', 'SMILES', 'Density']]
data_rg = df_train.loc[~df_train['Rg'].isna(), ['id', 'SMILES', 'Rg']]


# class holding all information of the regressor we are creating in one place
class NWRegressor():
    def __init__(self, df_train, model, kernel_bandwidth):
        # store the model as an attribute
        self.model = model
        # store as an attribute, the bandwidth of the kernel
        self.bandwidth = kernel_bandwidth
        # the df_train is also an integral part of the object, so should be stored as an attribute
        self.df_train = df_train
        # store all the data-subset attributes
        self.data_tg = df_train.loc[~df_train['Tg'].isna(), ['id', 'SMILES', 'Tg']]
        self.data_ffv = df_train.loc[~df_train['FFV'].isna(), ['id', 'SMILES', 'FFV']]
        self.data_tc = df_train.loc[~df_train['Tc'].isna(), ['id', 'SMILES', 'Tc']]
        self.data_density = df_train.loc[~df_train['Density'].isna(), ['id', 'SMILES', 'Density']]
        self.data_rg = df_train.loc[~df_train['Rg'].isna(), ['id', 'SMILES', 'Rg']]

    # separate method to initialize and assign embeddings because sometimes we may have them already computed and just want to assign so there is no need to do this in __init__
    def store_embeddings(self):
        # store the embeddings of all the smiles in df_train
        self.embeddings = self.model.encode(self.df_train['SMILES']) # shape: (nrow(df_train), 1024) = (7973, 1024)

    def kernel_fn(self, x):
        numerator = -(x * x)
        denominator = 2 * self.bandwidth * self.bandwidth
        return np.exp(numerator/denominator)
    
    def distance(self, arr1, arr2):
        # Compute pairwise distances (Euclidean by default)
        arr1_exp = arr1[:, np.newaxis, :]  # (x, 1, 120)
        arr2_exp = arr2[np.newaxis, :, :]  # (1, y, 120)
        sq_diff = (arr1_exp - arr2_exp) ** 2  # (x, y, 120)
        sq_distances = np.sum(sq_diff, axis=2)  # (x, y)
        distances = np.sqrt(sq_distances)  # (x, y)
        return distances

    # data_property is the dataframe corresponding to a certain property for which predictions are being made
    def predict_property(self, test_embeddings, data_property):
        # find out the distances between test_embeddings and self.embeddings' subset of data_property
        dists = self.distance(self.embeddings[data_property.index], test_embeddings) # shape: (nrow(data_property), test_embeddings)
        weights = self.kernel_fn(dists) # pass the mere distances to kernel to get weights
        # now, we scale the kernel-passed weights so that for each test_embedding, the weights' sum is 1
        weights_scaled = weights/weights.sum(axis = 0)
        # now we just tranpose as test_embeddings being as rownames are more intuitive
        weights_scaled = weights_scaled.T # shape: (test_embeddings,nrow(data_property))
        # now, just perform dot product with tg values vector to get tg value for each test_embedding
        train_tg_values = data_property.iloc[:, -1].values
        output_property_values = np.matmul(weights_scaled, train_tg_values)
        return output_property_values # shape: (test_embeddings,)

        # HERE... go through the details of the nadaraya-watson implementation. Where scaling, softmax etc. Then, finally
        # use the scaled weights to compute the avg value of tg.
    
    # predict on a list of smiles
    def predict(self, smiles):
        # create empty numpy arrays on which we can append as we get more and more predictions
        tg = np.empty(0)
        ffv = np.empty(0)
        tc = np.empty(0)
        density = np.empty(0)
        rg = np.empty(0)
        # loop is for being able to treat big list of smiles in parts
        for i in range(0, len(smiles), 1000):
            smiles_subset = smiles[i:i+1000]
            test_embeddings = self.model.encode(smiles_subset)
            # call the predict method for each property and return the values
            tg = np.concatenate([tg, self.predict_property(test_embeddings, self.data_tg)])
            ffv = np.concatenate([ffv, self.predict_property(test_embeddings, self.data_ffv)])
            tc = np.concatenate([tc, self.predict_property(test_embeddings, self.data_tc)])
            density = np.concatenate([density, self.predict_property(test_embeddings, self.data_density)])
            rg = np.concatenate([rg, self.predict_property(test_embeddings, self.data_rg)])
        
        return {'Tg': tg, 'FFV': ffv, 'Tc': tc, 'Density': density, 'Rg': rg}

%time embeddings = model.encode(df_train['SMILES'])

# finding good bandwidth using validation
from sklearn.model_selection import train_test_split
df_train_subset1, df_train_subset2 = train_test_split(df_train, test_size=0.05, train_size=0.95)

# resetting indices is important as the NWRegressor class assumes the df to have index in proper order
df_train_subset1 = df_train_subset1.reset_index(drop = True)
df_train_subset2 = df_train_subset2.reset_index(drop = True)

%time embeddings_df_train_subset1 = model.encode(list(df_train_subset1['SMILES']))

# loop through a range of bandwidth values and get the mse for each of them
for bw in [0.1, 0.5, 1, 3, 5, 7, 10, 15, 20]:
	# bw = 1
    regressor = NWRegressor(df_train_subset1, model, bw)
    regressor.embeddings = embeddings_df_train_subset1
    results = regressor.predict(list(df_train_subset2['SMILES']))
    mse = MSE(results, df_train_subset2)
    print(f'MSE for bandwidth {bw} is: {mse}')














