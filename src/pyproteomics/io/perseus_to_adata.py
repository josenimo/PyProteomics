import os
import sys
import pandas as pd
import numpy as np
import anndata as ad
import time
from loguru import logger

datetime = time.strftime("%Y%m%d_%H%M%S")

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}")

def perseus_to_anndata(path_to_perseus_txt):
    from perseuspy import pd
    df = pd.read_perseus(path_to_perseus_txt)
    # get obs headers
    obs_headers = list(df.columns.names)
    # get obs contents
    obs = [col for col in df.columns.values] #tuples
    obs = pd.DataFrame(obs)
    # var headers HARDCODED
    var_headers = obs.iloc[-4:,0].values.tolist()
    #remove rows with empty strings
    obs = obs[obs != '']
    obs.dropna(inplace=True)
    #rename headers
    obs.columns = obs_headers
    #var 
    var = df[var_headers]
    var.columns = var_headers
    #get data
    data = df.iloc[:,:-(len(var_headers))].values.T
    adata = ad.AnnData(X=data, obs=obs, var=var)
    return adata