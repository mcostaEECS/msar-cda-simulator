from __future__ import division

# --- Core Python --- #
import os
import re
import time
import array
import datetime

# --- NumPy & SciPy --- #
import numpy as np
from numpy import zeros, sqrt, mean, linspace, concatenate, cumsum
import scipy.io as sio
from scipy.stats import norm
from scipy.spatial.distance import cdist

# --- Image and Raster Processing --- #
import cv2
from PIL import Image
from skimage.morphology import disk
from skimage.filters.rank import median
import rasterio
from rasterio.plot import show, show_hist
from osgeo import gdal

# --- Plotting --- #
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import scienceplots

# --- Progress Bar --- #
import pyprind

# Custom module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.load_sample import load_sample
from tools.MSARkSample import MSARkSample


#while solving the problems

#------------------------- Split Image ------------------------#
def subarrays(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
def moving_average(im, K):
    kernel1 = np.ones((K,K),np.float32)/(K**2)
    Filtered_data = cv2.filter2D(src=im, ddepth=-1, kernel=kernel1) 
    return Filtered_data
    
  

    

def cdaTest(test_type, h, N, K):
    
    if h=='noise' or h=='correct':
        
        par=load_sample(test_type)[2]
        s=2
        
    elif h == 'free':
        
        par=load_sample(test_type)[1]
        s=1
        
        
        
    tp=par[23]; tp =  np.fliplr(tp); TP =par[24]
    
    start1=0; end1 = 500; start2=0; end2 = 500
    
    Itest = par[25][start1:end1,start2:end2]
    
    ItestSub = subarrays(Itest, N, N)
   
    IrefA = subarrays(par[0][start1:end1,start2:end2],N,N)
    IrefB = subarrays(par[1][start1:end1,start2:end2],N,N)
    IrefC = subarrays(par[2][start1:end1,start2:end2],N,N)
    IrefD = subarrays(par[3][start1:end1,start2:end2],N,N)
    IrefE = subarrays(par[4][start1:end1,start2:end2],N,N)
    IrefF = subarrays(par[5][start1:end1,start2:end2],N,N)
    IrefG = subarrays(par[6][start1:end1,start2:end2],N,N)
    IrefH = subarrays(par[7][start1:end1,start2:end2],N,N)
    IrefI = subarrays(par[8][start1:end1,start2:end2],N,N)
    IrefJ = subarrays(par[9][start1:end1,start2:end2],N,N)
    IrefK = subarrays(par[10][start1:end1,start2:end2],N,N)
    IrefL = subarrays(par[11][start1:end1,start2:end2],N,N)
    IrefM = subarrays(par[12][start1:end1,start2:end2],N,N)
    IrefN = subarrays(par[13][start1:end1,start2:end2],N,N)
    IrefO = subarrays(par[14][start1:end1,start2:end2],N,N)
    IrefP = subarrays(par[15][start1:end1,start2:end2],N,N)
    IrefQ = subarrays(par[16][start1:end1,start2:end2],N,N)
    IrefR = subarrays(par[17][start1:end1,start2:end2],N,N)
    
    IrefS = subarrays(par[18][start1:end1,start2:end2],N,N)
    IrefT = subarrays(par[19][start1:end1,start2:end2],N,N)
    IrefU = subarrays(par[20][start1:end1,start2:end2],N,N)
    IrefV = subarrays(par[21][start1:end1,start2:end2],N,N)
    IrefX = subarrays(par[22][start1:end1,start2:end2],N,N)
    
   
    p = 2 # C-C
    
    Data = [ItestSub[p], IrefA[p], IrefB[p], IrefC[p], IrefD[p], IrefE[p], IrefF[p], IrefG[p],
                    IrefH[p],IrefI[p], IrefJ[p], IrefK[p], IrefL[p], IrefM[p], IrefN[p], IrefO[p], 
                    IrefP[p], IrefQ[p],IrefR[p], IrefS[p], IrefT[p], IrefU[p], IrefV[p], IrefX[p], 
                    TP, N, K,s, test_type, h]
                
    resAvgCC = MSARkSample(Data)[0]
    resSupCC = MSARkSample(Data)[1]
    
    p = 10 # C-T
    
    Data = [ItestSub[p], IrefA[p], IrefB[p], IrefC[p], IrefD[p], IrefE[p], IrefF[p], IrefG[p],
                    IrefH[p],IrefI[p], IrefJ[p], IrefK[p], IrefL[p], IrefM[p], IrefN[p], IrefO[p], 
                    IrefP[p], IrefQ[p],IrefR[p], IrefS[p], IrefT[p], IrefU[p], IrefV[p], IrefX[p], 
                    TP, N, K,s, test_type, h]
                
    resAvgCT = MSARkSample(Data)[0]
    resSupCT = MSARkSample(Data)[1]
    upper_bound = MSARkSample(Data)[2]
    
    return resSupCT.ravel(), resSupCC.ravel(), upper_bound
    
    
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



if __name__ == "__main__":

    test_type = ['MSAR', 'MSARk', 'MSARh', 'MSARkh']
    h_conditions = ['free', 'noise', 'correct']
    N = 125
    K = 9
    DPI = 300

    # Estilo dos plots
    plt.rc('text', usetex=True)
    SMALL_SIZE = 22
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=DPI)
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    with plt.style.context(['science', 'ieee', 'std-colors']):
        for j, h in enumerate(h_conditions):
            # --- Linha 1: MSAR, MSAR, MSARh ---
            if h in ['free', 'noise']:
                Test_upper = test_type[0]  # MSAR
            else:
                Test_upper = test_type[2]  # MSARh

            ct_label = r'$C\to T$' if h == 'free' else r'$C_{\textnormal{int}}\to T_{\textnormal{int}}$' if h == 'noise' else r'$C_{\textnormal{int}}^{h}\to T_{\textnormal{int}}^{h}$'
            cc_label = r'$C\to C$' if h == 'free' else r'$C_{\textnormal{int}}\to C_{\textnormal{int}}$' if h == 'noise' else r'$C_{\textnormal{int}}^{h}\to C_{\textnormal{int}}^{h}$'

            res_CT, res_CC, _ = cdaTest(Test_upper, h, N, K)
            axes[0][j].plot(res_CT[:5000], color='rosybrown', linewidth=2, alpha=0.7, label=ct_label)
            axes[0][j].plot(res_CC[:5000], color='slategray', linewidth=2, alpha=0.9, label=cc_label)
            axes[0][j].set_title(f'{Test_upper} | h = {h}', fontsize=20)
            axes[0][j].legend(prop={'size': 12})
            axes[0][j].set_ylabel('Magnitude', fontsize=16)
            axes[0][j].set_xlabel(r'pixels [$1\times N$]', fontsize=16)

            # --- Linha 2: MSARk, MSARk, MSARkh ---
            Test_lower = test_type[1] if h in ['free', 'noise'] else test_type[3]

            res_CT, res_CC, _ = cdaTest(Test_lower, h, N, K)
            axes[1][j].plot(res_CT[:5000], color='rosybrown', linewidth=2, alpha=0.7, label=ct_label)
            axes[1][j].plot(res_CC[:5000], color='slategray', linewidth=2, alpha=0.9, label=cc_label)
            axes[1][j].set_title(f'{Test_lower} | h = {h}', fontsize=20)
            axes[1][j].legend(prop={'size': 12})
            axes[1][j].set_ylabel('Magnitude', fontsize=16)
            axes[1][j].set_xlabel(r'pixels [$1\times N$]', fontsize=16)

    fig.suptitle(r'MSAR Variants Comparison under Clutter Conditions', fontsize=24, y=0.98)

    path = 'results/plots/'
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'Vector_MSAR_2x3_corrected.png'), bbox_inches='tight', dpi=DPI)

    plt.show()

