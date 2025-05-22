import os
import pandas as pd
from datetime import datetime

def adata_to_perseus(adata, path_to_dir, suffix, obs_key=None):
    os.makedirs(path_to_dir, exist_ok=True)  # Ensure output directory exists

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Prepare file paths
    data_file = os.path.join(path_to_dir, f"{timestamp}_data_{suffix}.txt")
    metadata_file = os.path.join(path_to_dir, f"{timestamp}_metadata_{suffix}.txt")

    # Export expression data
    df = pd.DataFrame(data=adata.X, columns=adata.var_names, index=adata.obs[obs_key])
    df.index.name = "Name"  # Perseus requires this
    df.to_csv(data_file, sep="\t")

    # Export metadata
    metadata = adata.obs.copy()
    metadata.set_index(obs_key, inplace=True)
    metadata.index.name = "Name"
    metadata.to_csv(metadata_file, sep="\t")

    print(f"Success: files saved as\n- {data_file}\n- {metadata_file}")