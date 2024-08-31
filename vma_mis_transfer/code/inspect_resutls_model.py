import time
import os as os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import LinearConstraint
import multiprocessing as mp
from matplotlib import rc
import matplotlib as mpl
from scipy.stats import norm
from util_func import *

# mpl.use('pgf')
# plt.rcParams.update({
#     "pgf.texsystem":
#     # "xelatex",
#     "lualatex",
#     "font.family":
#     "serif",
#     "text.usetex":
#     False,
#     "pgf.rcfonts":
#     False,
#     "pgf.preamble": [
#         "\\usepackage{fontspec}", "\\usepackage{amsmath,amsfonts,amssymb}",
#         "\\usepackage{gensymb}", r"\setmainfont{Arial}"
#     ]
# })

if __name__ == '__main__':

    # fit_state_space_with_g_func_2_state_plus_noise()
    # fit_state_space_with_g_func_2_state_boot_with_noise()
    # inspect_results_boot_with_noise()

    # fit_state_space_with_g_func_2_state()
    fit_state_space_with_g_func_2_state_boot()
    # inspect_results_boot()
    # inspect_boot_stats()
