import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor

# mse between all the values we have in df_train_subset2, vs what we predicted
def MSE(results, df_train_subset2):
    # now, for all the values we did have in df_train_subset2, we will see how different our predictions were from them
    mask = df_train_subset2.iloc[:, -5:].isna().values
    # convert results to array
    results_arr = pd.DataFrame(results).values
    diff = results_arr - df_train_subset2.iloc[:, -5:].values
    mse = (diff[~mask]*diff[~mask]).mean()
    return mse

# Compute custom loss (wMAE) between predicted values of properties and their actual values.
def CustomLoss(y_hat_dict, y_dict):
    """
    Compute custom loss (wMAE) between predicted values of properties and their actual values.

    Args:
        - y_hat_dict: dict with keys as properties names ('Tg', 'FFV', 'Tc', 'Density', 'Rg')
        and values as their prediction values by whatever model was used.
        - y_dict: dict with keys as properties names ('Tg', 'FFV', 'Tc', 'Density', 'Rg')
        and values as their actual values in the data.

    Return:
        The wMAE loss as defined in the kaggle contest.
    """

    # compute wi for each property according to available data
    wi_dict = ComputeWiOfProperties(y_dict)

    # get the names of the properties
    props = list(y_dict.keys())

    # convert the dicts to 2d np.arrays
    y_array = np.array(list(y_dict.values())).T
    y_hat_array = np.array(list(y_hat_dict.values())).T
    wi_array = np.array(list(wi_dict.values())) # no need to ensure exact shape match as broadcasting takes care of that
    
    # now compute the loss
    inner_sum_vec = np.nansum(wi_array * np.abs(y_hat_array - y_array), axis = 1)
    loss = np.mean(inner_sum_vec)

    return loss

# compute wi term (used in computing the loss given predictions) for the properties.
def ComputeWiOfProperties(test_data_dict):
    # get the names of the properties
    props = test_data_dict.keys()
    K = len(props) # K is just the number of properties we are dealing with

    # if not already, convert the type of each value in the input dict to np.ndarray
    for prop in props:
        if type(test_data_dict[prop]) != np.ndarray:
            test_data_dict[prop] = np.array(test_data_dict[prop])

    # make ni dict storing ni values for all properties
    ni_dict = {}
    for prop in props:
        ni_dict[prop] = np.sum(~np.isnan(test_data_dict[prop]))

    # make ri dict storing ri values for all properties
    ri_dict = {}
    for prop in props:
        ri_dict[prop] = np.nanmax(test_data_dict[prop]) - np.nanmin(test_data_dict[prop])

    # finally, make the wi dict storing the wi values for all properties
    wi_dict = {}
    for prop in props:
        term1 = 1/ri_dict[prop]
        term2_num = K*np.sqrt(1/ni_dict[prop])
        term2_den = sum([np.sqrt(1/i) for i in ni_dict.values()])
        term2 = term2_num/term2_den
        wi_dict[prop] = term1*term2

    return wi_dict

# Perform cross-validation on a model, given a data
# this function takes in the model and data for a specific property only.
def CrossValidate(model, X, y, num_cv = 5):

    # initialize the cv object which stores information about how the split is done
    cv = KFold(n_splits = num_cv, shuffle = True, random_state = 42)

    # get the scores of the model for each cv fold
    scores = cross_val_score(model, X, y, cv = cv, scoring = 'neg_mean_absolute_error')

    # return the mean of scores
    return np.round(np.mean(scores), 4)

# take X and y, and param_grid, and optimize hyperparameters of random forest accordingly and 
# return the best estimator
# number of fits will be n_iter*5 (as cv I have kept to be constant, 5)
def OptimizeHyperparamsRF(X, y, n_iter = 10):
    # since I don't have an intuition as to what set of params may be best for each 
    # I am using same set of param values for each parameter
    param_distributions = {
        'n_estimators': randint(low=50, high=200),  # Number of trees in the forest
        'max_depth': randint(low=5, high=20),       # Maximum depth of the tree
        'min_samples_split': randint(low=2, high=10), # Minimum number of samples required to split an internal node
        'min_samples_leaf': randint(low=1, high=5),  # Minimum number of samples required to be at a leaf node
    }
    model = RandomForestRegressor()
    random_search = RandomizedSearchCV(estimator = model, param_distributions = param_distributions, 
                                       n_iter = n_iter, cv = 5, scoring = 'neg_mean_absolute_error', 
                                       random_state = 42, verbose = 2)
    random_search.fit(X, y)
    print(f'Best score: {random_search.best_score_}')
    print(f'Best params: {random_search.best_params_}')
    return random_search.best_estimator_

# function to train a random forest model, given the params, and the original df will all 
# information and property name.
# model params is a dict storing the keyword arguments for RandomForestRegressor
# data (which is a df) should have the embedding col
def TrainRFModel(models_params_dict, data, prop):
    # first, get the subset of data using prop
    data_subset = data.loc[~data[prop].isna(), ['id', 'SMILES', prop, 'embedding']]

    # then get the X and y values
    X = np.vstack(data_subset['embedding'])
    y = data_subset[prop]
    print(f'--> Property: {prop}')
    print('Gotten the appropriate (subsetted to non nan values) X and y from data')

    # Init and fit the model
    model = RandomForestRegressor(**models_params_dict[prop])
    print('Fitting the model now')
    model.fit(X, y)
    print(f'Fitted the model!')
    return model

# generate the prediction for all the properties for each datapoint using the dict of trained 
# models and embeddings for the smiles
def GeneratePredictions(embeddings, models_dict):
    # use each model to generate predictions and return a dict of predicted values for 
    # each property
    y_hat_dict = {}
    for prop in models_dict.keys():
        y_hat = models_dict[prop].predict(embeddings)
        y_hat_dict[prop] = y_hat
    return y_hat_dict

# generate embeddings of smiles in batches instead of all at once to save ram
def GenerateEmbeddingsInBatches(smiles_list, st_model, batch_size = 1000):
    embeddings = np.empty(shape = (len(smiles_list), 384)) # 384 is according to the model on kaggle
    # loop through the smiles in batches, generate the batch's embeddings and accordingly, modify the 
    # entries in embeddings np array
    for i in range(0, len(smiles_list), batch_size):
        smiles_subset = smiles_list[i:i+batch_size]
        embeddings[i:i+batch_size, :] = st_model.encode(smiles_subset)
    return embeddings

# generate the output csv from the predictions dict and the df_test dataframe
def SaveOutputCSV(df_test, preds_dict, output_file):
    # first, take the df_test df and remove SMILES col from it
    df_test.drop(labels='SMILES', axis=1, inplace=True)

    # next, pd.concat (horizontally) to it, the df of the predictions
    output_df = pd.concat([df_test, pd.DataFrame(preds_dict)], axis = 1)
    
    # save without index
    output_df.to_csv(output_file, index = False)
    return
