import os
import argparse
import subprocess
import zipfile

class KaggleException(Exception):
    pass

def download_kaggle(dir: str, competition: str):
    """ 
    Function that downloads the given competition string
    into a local directory.

    input: directory - string, competition - string

    output: none

    ex. download_kaggle("../data/raw/", "neurips-open-polymer-prediction-2025")
    """


    try:
        # Download the kaggle competition using the kaggle api.
        subprocess.run(['kaggle', 'competitions', 'download','-c' f'{competition}'])

        # Check to see if the path exists.
        if not os.path.exists(dir):
            raise KaggleException("Directory does not exist")
        # If it does exist we move the zip file into the directory.
        else:

            subprocess.run(['mv', f'{competition}.zip', f'{dir}'])
            # Now unzip the file. Credit to https://stackoverflow.com/questions/3451111/unzipping-files-in-python.
            with zipfile.ZipFile(os.path.join(dir, f'{competition}.zip')) as zipref:
                zipref.extractall(dir)  # Raise a custom exception if anything occurs.
            
            # Now we remove the original zip file.
            subprocess.run(['rm', f'{dir}/{competition}.zip'])

    except Exception as e:
        raise KaggleException(e)

    return


def main():
    """ Main function """
    
    # Create arpgarser with arguments directory and competition.
    parser = argparse.ArgumentParser(
                prog = "Retrieval of data",
                description = "Retrieves the given kaggle dataset, essentially a wrapper around kaggle api for the dataset download",
            )
    parser.add_argument("-d", "--directory", required = True)
    parser.add_argument("-c", "--competition", required = True)
    args = parser.parse_args()

    # Check to see if kaggle path exists.
    if os.path.exists(os.path.join(args.directory, args.competition)):
        pass
    # Else we download the competition to the directory.
    else:
        download_kaggle(args.directory, args.competition)

    return

if __name__ == "__main__":
    main()
        
