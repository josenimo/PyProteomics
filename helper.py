import pandas as pd
import numpy as np
import tabulate



def DIANN_to_adata(DIANN_path:str, metadata_path:str, metadata_check=False):

    """
    Created by Jose Nimo on 19.01.2024

    Description:
    Converts DIANN output file and metadata file into anndata object.
    Assumes DIANN output file is tab-delimited, with the first 5 columns being metadata, and the rest being protein expression data.
    Assumes metadata file is tab-delimited, with a column of sample names with columnd header called 'Name', and the rest being metadata.

    Arguments:
    DIANN_path: path to DIANN output file
    metadata_path: path to metadata file
    metadata_check: boolean, if True, prints metadata values

    """
    
    print("Step 1: Loading DIANN output file")
    #load DIANN output file
    df = pd.read_csv(DIANN_path, sep="\t")
    #all rows, all columns except first 5 to remove metadata
    rawdata = df.iloc[:,5:]
    rawdata = rawdata.transpose() #transpose to have samples as rows and proteins as columns

    print("Step 2: Loading metadata file")
    sample_metadata = pd.read_csv(metadata_path, sep="\t") #load metadata file
    sample_metadata.index = sample_metadata["Name"] #set index to be the sample name, matching the rawdata index
    sample_metadata = sample_metadata.drop("Name", axis=1) #drop the name column, since it is now the index

    #check if metadata number matches rawdata number
    if rawdata.shape[0] != sample_metadata.shape[0]:
        print("ERROR: Number of samples in DIANN output and metadata file do not match. Please check your files.")
        return None

    if metadata_check:
        #lets see the metadata values
        categorical_values_dict = {}
        for column in sample_metadata.columns:
            categorical_values_dict[column] = sample_metadata[column].unique().tolist()
        # Convert the dictionary into a list of tuples
        data_list = [(key, value) for key, value in categorical_values_dict.items()]
        # Print the dictionary in a tabular format
        print("Sample Metadata")
        print(tabulate(data_list, headers=["Column Name", "Unique Values"], tablefmt="grid"))
        print("\n")

    print("Step 3: Loading protein metadata")
    protein_metadata = df.iloc[:,:5] #protein metadata
    protein_metadata.index = protein_metadata["Protein.Group"] #set index to be the protein name, matching the rawdata index
    protein_metadata = protein_metadata.drop("Protein.Group", axis=1) #drop the name column, since it is now the index
    print(f"For a total of {protein_metadata.shape[0]} proteins \n")

    print("Step 4: Creating anndata object")
    adata = ad.AnnData(X=rawdata.values, obs=sample_metadata, var= protein_metadata) #create anndata object
    print(adata)
    print("\n")

    return adata
