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
    
    
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
    
      

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
    
    

    
 
    with plt.style.context(['science', 'ieee', 'std-colors']):
            
            fig, ax1 = plt.subplots(1, 1, figsize=(8,8), dpi=DPI) #, constrained_layout=True)
            ax1.set_prop_cycle(color=['rosybrown', 'slategray', 'red', 'gray']) #linewidth=2
            
            h = h[0] # 0,1,2
            
            if h=='free':
                ct = r'$C\to T$'
                cc = r'$C\to C$'
                
            elif h=='noise':
                ct = r'$C_{\textnormal{int}}\to T_{\textnormal{int}}$'
                cc = r'$C_{\textnormal{int}} \to C_{\textnormal{int}}$'
                
            elif h=='correct':
                ct = r'$C_{\textnormal{int}}^{h}\to T_{\textnormal{int}}^{h}$'
                cc = r'$C_{\textnormal{int}}^{h} \to C_{\textnormal{int}}^{h}$'
                
            
            Test = test_type[0] # h=0 [Test=0,1]; h=1 [Test=0,1]; h=2 [Test=2,3]
            
            ax1.plot(cdaTest(Test, h, 125, K)[0][0:5000], color='rosybrown', linewidth=2,  alpha=0.6, label=ct)
            ax1.plot(cdaTest(Test, h, 125, K)[1][0:5000], color='slategray', linewidth=2,  alpha=0.8,label=cc)
    
            ax1.legend(prop={'size': 20})
        
            #ax1.invert_xaxis()
            ax1.set_ylabel(r'Magnitue', fontsize=30)  
            ax1.set_xlabel(r'pixels [$1\times N$]', fontsize=30)  
            ax1.xaxis.set_label_coords(.5, -0.085)
            ax1.yaxis.set_label_coords(-0.10, 0.5)

    path='results/plots/'
    namefile = 'Vector_%s_Int_%s.png'%(Test,h)
 
    fig.savefig(path + namefile, dpi=DPI)
    #fig.tight_layout()
            
          

    plt.show()      

            
        
    
 












