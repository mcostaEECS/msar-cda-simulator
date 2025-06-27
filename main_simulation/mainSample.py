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

from tools.load_sample import load_sample
from tools.MSARkSample import MSARkSample


#------------------------- Split Image ------------------------#
def subarrays(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def msarSim(test_type, N, K, status):
    
    if status == 'free':
        s= 1
    elif status== 'noise':
        s = 2
        
    
      
    campaign = 'Runtime_%s_N_%d_K_%d'%(test_type, N, K)
    
    campaignRuntime = 'Runtime_%s_N_%d_K_%d'%(test_type, N, K)
    campaignPower = '%s_N_%d_K_%d'%(test_type, N, K)
    
    path = 'results/logs/sample/'

    nroPairs = 1
    dimH =500; dimW = 500
  
            
    TEST_time=[]; CFAR_time = []; Nop=[]
    bar = pyprind.ProgBar(nroPairs, monitor=True, title=campaign)
    
    par=load_sample(test_type)[s]
    tp=par[23]; tp =  np.fliplr(tp); TP =par[24]
    pair = s 
        
    campaign2 = 'ICD_%s_N_%d_K_%d_tp_%s_pair_%d'%(test_type,N,K,TP, pair)
    start1=0; end1 = 500; start2=0; end2 = 500
            
    Itest = par[25][start1:end1,start2:end2]  
    Itest = subarrays(Itest, N, N)
        
        
    name_file = path+campaign+'.txt'
    with open(name_file, 'a') as f:
        
        if test_type == 'MSAR':
            pathOut = 'results/dataMSAR/dataMSAR_sample/'
            lags = 11
        elif test_type == 'MSARk':
            pathOut = 'results/dataMSARk/dataMSARk_sample/'
            lags = 23
        elif test_type == 'MSARkh':
            pathOut = 'results/dataMSARkh/dataMSARkh_sample/'
            lags = 23
        
            
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
                
        #------------------------ Parallel Processing  -------------------------#
        START = time.time()
        cnt = 0
        CDn = []
        for k in range(len(IrefParallel)):
            set_start_method('fork', force=True)
            try:
                pool = Pool(8) 
                CD = [r[0] for r in pool.starmap(MSARkSample, zip(IrefParallel[k]))]

                #CD = pool.starmap(MSARkSample, zip(IrefParallel[k])) 
                
                
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
        
        
        
        an = time_cfar
        ar = time_cfar/lags
        
        CDout = []
        for a in range(len(CDn)):
            for b in range(len(CDn[a])):
                CDout.append(CDn[a][b])

            
        ICD=[]
        im = []; W = 500
        for j in range(0,len(CDout),int(W/N)):
            a=[]
            for l in range(int(W/N)):
                a.append(CDout[j+l])
            im.append(np.hstack((a)))
        ICD = np.vstack((im))
        name_file = pathOut+campaign2+'.mat'  
        scipy.io.savemat(name_file, {'ICD':ICD})
                        
        f.write(str(test_type)+';'+'N_'+str(N)+';'+'K_'+str(K)+';'+'pair_'+str(s)+';'+'runtime_per_regime_'+str(np.sum(ar))+';'+'runtime_total_'+str(np.sum(an))+'\n')
        bar.update()
            
        TEST_time.append(np.sum(ar))
        CFAR_time.append(np.sum(an))



if __name__ == "__main__":
    
    #'MSAR' = (k=11, d=5,\ell=1,h=0) 
    #'MSARk' = (k=23,d=5,\ell=2,h=0)
    #'MSARk' = (k=23,d=5,\ell=2,h=1)


     
    test_type = ['MSAR', 'MSARk', 'MSARkh']
    s = ['free', 'noise']
    test_type  = test_type[1]
    K = 9
    N = 125
    msarSim(test_type, N, K, s[1])
        
    