""" Stratifies learning data into n number of folds as a CLI argument """
import os
import argparse
from sklearn.model_selection import KFold
import pandas as pd


class SplittingException(Exception):
    """Exception for the splitting cli script"""
    pass

def main(file:str,
         folds: int,
         outdir: str):
    """ Main function """
    # Check to make sure passed argument file directories
    # actually exist.
    if not os.path.exists(file) or not os.path.exists(outdir):
        raise SplittingException()

    try:
        # Read in file --- MUST BE CSV FILE -- into memory
        # maybe a bad choice if data is bigger...
        df = pd.read_csv(file)

        # Create the fold column..
        df['fold'] = -1
        
        # Create n folds.
        folds = KFold(n_splits=folds, shuffle=True, random_state=42)
        for fold_idx, (_, val_idx) in enumerate(folds.split(df)):
            df.loc[val_idx, "fold"] = fold_idx

        # Write out dataframe.
        df.to_csv(os.path.join(outdir, "folds.csv"), index=False)
        print(f'Saved dataframe with {folds} folds to {outdir}.')
        return
    except Exception as e:
        raise SplittingException(e)

if __name__ == "__main__":
    # Create argument parser.
    parser = argparse.ArgumentParser(
        prog='Splitter',
        description='Stratifies learning data into n folds')
    
    # Create arguments.
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-n", "--folds", required=True)
    parser.add_argument("-o", "--outdir", required=True)

    args = parser.parse_args()
    main(args.file, int(args.folds), args.outdir)
