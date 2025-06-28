"""
Main simulation script for MSAR-based change detection algorithms.

This script loads multitemporal image datasets with directional diversity,
applies MS-AR(k,d,ℓ,h) models, and computes change detection maps (ICD)
using multiprocessing. The outputs include .mat files with ICD results
and runtime logs for performance assessment.

Author: Marcello G. Costa  
Institution: ITA / KTH (Visiting Researcher)  
Date: 2025-04-22
"""
from __future__ import division

# Core libraries
import numpy as np
import time
import os
import subprocess
import re
import signal
import datetime
import psutil

# Data handling
import scipy.io
from scipy.stats import norm

# Progress bar
import pyprind

# Multiprocessing
from torch.multiprocessing import Pool, set_start_method, freeze_support
from concurrent.futures import ProcessPoolExecutor


# Custom module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.load_data import load_data
from tools.MSARk import MSARk



#------------------------- Split Image ------------------------#
def subarrays(arr, nrows, ncols):
    """
    Splits a 2D image array into non-overlapping subarrays.

    Parameters:
        arr (ndarray): Input 2D image.
        nrows (int): Number of rows per subarray.
        ncols (int): Number of columns per subarray.

    Returns:
        ndarray: Reshaped array with shape (blocks, nrows, ncols)
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    

def msarSim(test_type, N, K):
    
    # Define campaign naming for output logs and runtime tracking
    campaign = 'Runtime_%s_N_%d_K_%d'%(test_type, N, K)
    
    path = 'results/logs/'

    nroPairs = 24 # Set to 1 for testing, or 24 for full campaign
    dimH = 3000; dimW = 2000
    
    
  
            
    TEST_time=[]; CFAR_time = []; Nop=[]
    bar = pyprind.ProgBar(nroPairs, monitor=True, title=campaign)
    for s in range(nroPairs):
        # Uncomment the following line for full campaign:
        # s = s + 15  # For manual testing on a specific pair (15)
        
        # Load dataset for a specific pair
        par=load_data(test_type)[s]
        tp=par[23]; tp =  np.fliplr(tp); TP =par[24]
        pair = s 
        
        
        # ======================= Configuration ======================= #
                
        campaign2 = 'ICD_%s_N_%d_K_%d_tp_%s_pair_%d'%(test_type,N,K,TP, pair)
        start1=0; end1 = 3000; start2=0; end2 = 2000
            
        Itest = par[25][start1:end1,start2:end2]  
        Itest = subarrays(Itest, N, N)
        
        
        name_file = path+campaign+'.txt'
        with open(name_file, 'a') as f:
            
            if test_type == 'MSAR':
                pathOut = 'results/dataMSAR/'
                lags = 11
            elif test_type == 'MSARk':
                pathOut = 'results/dataMSARk/'
                lags = 23
            elif test_type == 'MSARkh':
                pathOut = 'results/dataMSARkh/'
                lags = 23
           
        # =================== Load and Preprocess Data ================= #

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
            
            
            step = 8
            IrefParallel = []
            for m in range(0,len(Itest),step):
                a=[]
                for n in range(step):
                    TestPairs = [Itest[n+m], IrefA[n+m], IrefB[n+m], IrefC[n+m], IrefD[n+m], IrefE[n+m], IrefF[n+m], IrefG[n+m], IrefH[n+m],
                                    IrefI[n+m], IrefJ[n+m], IrefK[n+m], IrefL[n+m], IrefM[n+m], IrefN[n+m], IrefO[n+m], IrefP[n+m], IrefQ[n+m],
                                    IrefR[n+m], IrefS[n+m], IrefT[n+m], IrefU[n+m], IrefV[n+m], IrefX[n+m], TP, N, K,s, test_type]
                    a.append(TestPairs)
                IrefParallel.append(a)
                
                
        # ================== Parallel Inference Phase ================== #
                
            # ---- Parallel Processing Setup ---- #
            # For each set of 8 image blocks, process using multiprocessing
            # MSARk returns three outputs: [resAvg, resSup, upper_bound]
            # Here, we only store the primary change detection map: resAvg
            
            START = time.time()
            cnt = 0
            CDn = []
            for k in range(len(IrefParallel)):
                set_start_method('fork', force=True)
                try:
                    pool = Pool(8) 
                    CD = [r[0] for r in pool.starmap(MSARk, zip(IrefParallel[k]))]
                    CDn.append(CD)
                    cnt =+ cnt+1

                except Exception as e:
                    print('Main Pool Error: ', e)
                except KeyboardInterrupt:
                    exit()
                finally:
                    pool.terminate()
                    pool.close() 
                    pool.join() 

            END = time.time()
            time_cfar = (END - START)
            
            
            
            total_time = time_cfar
            avg_time_per_regime = time_cfar/lags
            
            CDout = []
            for a in range(len(CDn)):
                for b in range(len(CDn[a])):
                    CDout.append(CDn[a][b])
    
            # --- Reassemble full image from blocks ---
            # ICD is the final change detection map for the full image              
            ICD=[]
            im = []; W = 2000
            for j in range(0,len(CDout),int(W/N)):
                a=[]
                for l in range(int(W/N)):
                    a.append(CDout[j+l])
                im.append(np.hstack((a)))
            ICD = np.vstack((im))
            name_file = pathOut+campaign2+'.mat'  
            
        # =================== Save Results and Logs ==================== #
           
            # Save ICD result to .mat file
            scipy.io.savemat(name_file, {'ICD':ICD})
            
            # Log runtime results to file                            
            f.write(str(test_type)+';'+'N_'+str(N)+';'+'K_'+str(K)+';'+'pair_'+str(s)+';'+'runtime_per_regime_'+str(np.sum(avg_time_per_regime))+';'+'runtime_total_'+str(np.sum(total_time))+'\n')
            bar.update()
                
            TEST_time.append(np.sum(avg_time_per_regime))
            CFAR_time.append(np.sum(total_time))



if __name__ == "__main__":
    
# Configuration of MS-AR model:
    # MSAR    = (k=11, d=5, ℓ=1, h=0)
    # MSARk   = (k=23, d=5, ℓ=2, h=0)
    # MSARkh  = (k=23, d=5, ℓ=2, h=1)

    test_type = ['MSAR', 'MSARk', 'MSARkh']
    test_type = test_type[0]  # Select one model for simulation
    K = 9                     # Model order
    N = 250                   # Block size (NxN)
    
    msarSim(test_type, N, K)
    