# import os
# import sys
# from datetime import datetime
# from tabulate import tabulate
# import shutil
# import time
# import warnings

# import anndata as ad
# import numpy as np
# import pandas as pd
# import scanpy as sc
# sc.settings.verbosity = 1

# from sklearn.utils import shuffle
# import scipy

# #plotting
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import seaborn as sns
# from adjustText import adjust_text


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)


def density_plots(adata):
