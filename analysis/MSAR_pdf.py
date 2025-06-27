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
                    TP, N, K,s, test_type]
                
    resAvgCC = MSARkSample(Data)[0]
    resSupCC = MSARkSample(Data)[1]
    
    p = 10 # C-T
    
    Data = [ItestSub[p], IrefA[p], IrefB[p], IrefC[p], IrefD[p], IrefE[p], IrefF[p], IrefG[p],
                    IrefH[p],IrefI[p], IrefJ[p], IrefK[p], IrefL[p], IrefM[p], IrefN[p], IrefO[p], 
                    IrefP[p], IrefQ[p],IrefR[p], IrefS[p], IrefT[p], IrefU[p], IrefV[p], IrefX[p], 
                    TP, N, K,s, test_type]
                
    resAvgCT = MSARkSample(Data)[0]
    resSupCT = MSARkSample(Data)[1]
    upper_bound = MSARkSample(Data)[2]
    
    return resSupCT.ravel(), resSupCC.ravel(), upper_bound
    #resSupCT.ravel(), resSupCC.ravel(), upper_bound
    
    
    
      

if __name__ == "__main__":
    
    test_type = ['MSAR', 'MSARk', 'MSARh', 'MSARkh']
    h = ['free', 'noise', 'correct']
    block = [100, 125]; K=9
    
    
   
   
    
    
     
        
 # # #------------------------ Graphic Analysis -------------------------#
 
    plt.rc('text', usetex=True)
    DPI =300


    SMALL_SIZE = 30
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 30

    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    
    N = 100

    
 
    with plt.style.context(['science', 'ieee', 'std-colors']):
            
            fig, ax2 = plt.subplots(1, 1, figsize=(8,8), dpi=DPI) #, constrained_layout=True)
            ax2.set_prop_cycle(color=['rosybrown', 'slategray', 'red', 'gray']) #linewidth=2
            
            ct = [r'$T$', r'$T_{\textnormal{int}}$',r'$ T_{\textnormal{int}}^{h}$' ]
            cc = [r'$C$', r'$C_{\textnormal{int}}$', r'$C_{\textnormal{int}}^{h}$']
    
                
    
            
            meanColor = ['k', 'navy'] 
            ax2.set_prop_cycle(color=['slategray', 'rosybrown', 'red', 'gray']) #linewidth=2
            
            Test = 'MSARk'
            
            sns.kdeplot(np.array(cdaTest(Test, 'free', 125, K)[1][0:5000]),color='slategray', lw=2,  alpha=0.8,ax=ax2, label=cc[0]) 
            sns.kdeplot(np.array(cdaTest(Test, 'noise', 125, K)[1][0:5000]),color='slategray',lw=2, linestyle='dotted', alpha=0.8,ax=ax2, label=cc[2])  
            sns.kdeplot(np.array(cdaTest(Test+'h', 'correct', 125, K)[1][0:5000]),color='slategray', linestyle='dashdot', marker='o', alpha=0.25,ax=ax2, label=cc[1]) 
            
            sns.kdeplot(np.array(cdaTest(Test, 'free', 125, K)[0][0:5000][0:5000]), color='rosybrown', lw=2, alpha=0.8,ax=ax2, label=ct[0]) 
            sns.kdeplot(np.array(cdaTest(Test, 'noise', 125, K)[0][0:5000]),color='rosybrown', lw=2, linestyle='dotted', alpha=0.8,ax=ax2, label=ct[2]) 
            sns.kdeplot(np.array(cdaTest(Test+'h', 'correct', 125, K)[0][0:5000]),color='rosybrown', alpha=0.25,  marker='o', linestyle='dashdot',ax=ax2, label=ct[1]) 

            upperB= cdaTest(Test, 'free', 125, K)[2]

            ax2.axvline(upperB, 0,1, color='red', alpha=0.35, label=r'threshold=%.2f'%upperB) 
            
        
           
           
           
            ax2.legend(prop={'size': 20})
            ax2.set_ylabel(r'Density', fontsize=30)  
            ax2.yaxis.set_label_position("right")
            ax2.set_xlabel(r'Standardized value (z-score)', fontsize=30)  
            ax2.xaxis.set_label_coords(.5, -.085)
            ax2.yaxis.set_label_coords(-0.10, 0.5)
            ax2.set_xlim([0.005, upperB+0.025])
            ax2.yaxis.set_tick_params(labelleft=False, labelright=True)

    path='results/plots/'
    namefile = 'PDF_%s.png'%(Test)
 
    fig.savefig(path + namefile, dpi=DPI)
    #fig.tight_layout()
            
          

    plt.show()      

            
        
    
 

