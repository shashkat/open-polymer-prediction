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
        # Read in file --- MUST BE CSV FILE
        df = pd.read_csv(file)
        
        # Create n folds.
        folds = KFold(n_splits=folds)
        splits = folds.split(df)
        print(splits)

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
