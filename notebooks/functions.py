import pandas as pd


# mse between all the values we have in df_train_subset2, vs what we predicted
def MSE(results, df_train_subset2):
    # now, for all the values we did have in df_train_subset2, we will see how different our predictions were from them
    mask = df_train_subset2.iloc[:, -5:].isna().values
    # convert results to array
    results_arr = pd.DataFrame(results).values
    diff = results_arr - df_train_subset2.iloc[:, -5:].values
    mse = (diff[~mask]*diff[~mask]).mean()
    return mse