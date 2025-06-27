from __future__ import division

# Core numerical and scientific computing
import numpy as np
from numpy import (
    zeros, sqrt, mean, linspace, concatenate, cumsum, inf, hstack
)

# Scientific tools and IO
import scipy.io as sio
import scipy.stats as stats
from scipy.io import savemat
from scipy.spatial import distance
from statsmodels.distributions.empirical_distribution import ECDF

# Image processing
import cv2
from PIL import Image
from skimage import data
from skimage.morphology import disk
from skimage.filters import gaussian
from skimage.filters.rank import minimum, maximum, mean as rank_mean, median

# Plotting
from matplotlib import pyplot as plt

# Parallel processing
from concurrent.futures import ProcessPoolExecutor
from torch.multiprocessing import Pool, set_start_method, freeze_support

# Custom module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.load_sample import load_sample


# Debugging / pretty print
from pprint import pp



#------------------------- Spatial Filetring ------------------------#
def moving_average(im, K):
    kernel1 = np.ones((K,K),np.float32)/(K**2)
    Filtered_data = cv2.filter2D(src=im, ddepth=-1, kernel=kernel1) 
    return Filtered_data


#------------------------- AR(1) Internal Filter ------------------------#
def test(Itest, Iref, N,K):
    
 
     
    TestVec =Itest.ravel()
    RefVec = Iref.ravel()
    
    
    rho = np.corrcoef(TestVec,RefVec)[0][1]
    unrho = np.sqrt(1-rho**2)
 
    th = 1.2
    CD0 = []
    for i in range(len(TestVec)):
        
        
        if TestVec[i] > th*RefVec[i]:
            resTarget = TestVec[i]*rho + RefVec[i]*unrho # degree of persistence
        else:
            resTarget = TestVec[i]
    
        CD0.append(resTarget)
        
    return CD0
    


#------------------------- MS-AR(k,\ell,h) ------------------------#
def MSARkSample(Data):
    
    # [Diversity/Test Training MS-AR(D,\ell,h)]
    Itest = Data[0]
    
    IrefA = Data[1]; IrefB = Data[2]; IrefC = Data[3]; IrefD = Data[4]; IrefE = Data[5]; IrefF = Data[6]
    IrefG = Data[7]; IrefH = Data[8]; IrefI = Data[9]; IrefJ = Data[10]; IrefK = Data[11]; IrefL = Data[12]
    IrefM = Data[13]; IrefN = Data[14]; IrefO = Data[15]; IrefP = Data[16]; IrefQ = Data[17]; IrefR = Data[18]
    IrefS = Data[19]; IrefT = Data[20]; IrefU = Data[21]; IrefV = Data[22]; IrefX = Data[23];   

    tp = Data[24]; N = Data[25]; K = Data[26]; s = Data[27]
    test_type = Data[28]
    
    ar19 = test(Itest, IrefS, N,K)
    ar20 = test(Itest, IrefT, N,K) 
    ar21 = test(Itest, IrefU, N,K)
    ar22 = test(Itest, IrefV, N,K) 
    ar23 = test(Itest, IrefX, N,K) 
    
    
    
    p19 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar19).flatten())[0][1])
    p20 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar20).flatten())[0][1])
    p21 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar21).flatten())[0][1])
    p22 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar22).flatten())[0][1])
    p23 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar23).flatten())[0][1])
    

    pST =  [ p19,  p20, p21, p22, p23]                      
    arST = [ ar19, ar20, ar21, ar22, ar23] 
    
    
    IdxST = sorted(range(len(pST)), key=lambda sub: pST[sub], reverse=True)[:10] 
    sST = [ IrefS, IrefT, IrefU, IrefV, IrefX] 
    
 
    intratio = pST[IdxST[2]]/ pST[IdxST[0]]
    
    # [Lags Prediction for Clutter Supression MS-AR(k,\ell,h)]    
    
    # Robust Long-term (with Interference Mitigation)
    if intratio > 0.9 and test_type == 'MSARkh':
        Itest = sST[IdxST[2]]
        
        IrefA = Data[1]; IrefB = Data[2]; IrefC = Data[3]; IrefD = Data[4]; IrefE = Data[5]; IrefF = Data[6]
        IrefG = Data[7]; IrefH = Data[8]; IrefI = Data[9]; IrefJ = Data[10]; IrefK = Data[11]; IrefL = Data[12]
        IrefM = Data[13]; IrefN = Data[14]; IrefO = Data[15]; IrefP = Data[16]; IrefQ = Data[17]; IrefR = Data[18]
        IrefS = Data[19]; IrefT = Data[20]; IrefU = Data[21]; IrefV = Data[22]; IrefX = Data[23];   

        tp = Data[24]; N = Data[25]; K = Data[26]; s = Data[27]
        test_type = Data[28]

        
        
        
        ar19 = test(Itest, IrefS, N,K)
        ar20 = test(Itest, IrefT, N,K) 
        ar21 = test(Itest, IrefU, N,K)
        ar22 = test(Itest, IrefV, N,K) 
        ar23 = test(Itest, IrefX, N,K) 
        
        
        
        p19 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar19).flatten())[0][1])
        p20 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar20).flatten())[0][1])
        p21 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar21).flatten())[0][1])
        p22 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar22).flatten())[0][1])
        p23 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar23).flatten())[0][1])
        

        pST =  [ p19,  p20, p21, p22, p23]                      
        arST = [ ar19, ar20, ar21, ar22, ar23] 
        
        
        ar1 = test(Itest, IrefA, N, K)
        ar2 = test(Itest, IrefB, N, K) 
        ar3 = test(Itest, IrefC, N, K) 
        ar4 = test(Itest, IrefD, N, K) 
        ar5 = test(Itest, IrefE, N,K) 
        ar6 = test(Itest, IrefF, N,K) 
        
        p1 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar1).flatten())[0][1])
        p2 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar2).flatten())[0][1])
        p3 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar3).flatten())[0][1])
        p4 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar4).flatten())[0][1])
        p5 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar5).flatten())[0][1])
        p6 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar6).flatten())[0][1])
        
        ar7 = test(Itest, IrefG, N,K) 
        ar8 = test(Itest, IrefH, N,K)
        ar9 = test(Itest, IrefI, N, K) 
        ar10 = test(Itest, IrefJ, N, K) 
        ar11 = test(Itest, IrefK, N, K) 
        
        p7 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar7).flatten())[0][1])
        p8 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar8).flatten())[0][1])
        p9 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar9).flatten())[0][1])
        p10 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar10).flatten())[0][1])
        p11 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar11).flatten())[0][1])
        
        ar12 = test(Itest, IrefL, N, K) 
        ar13 = test(Itest, IrefM, N,K) 
        ar14 = test(Itest, IrefN, N,K) 
        ar15 = test(Itest, IrefO, N,K) 
        ar16 = test(Itest, IrefP, N,K) 
        ar17 = test(Itest, IrefQ, N,K) 
        ar18 = test(Itest, IrefR, N,K)
        
        p12 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar12).flatten())[0][1])
        p13 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar13).flatten())[0][1])
        p14 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar14).flatten())[0][1])
        p15 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar15).flatten())[0][1])
        p16 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar16).flatten())[0][1])
        
        p17 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar17).flatten())[0][1])
        p18 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar18).flatten())[0][1])
        
        pLT = [p1,p2,p3,p4, p5, p6, p7, p8, p9,p10,p11,p12, p13, p14, p15, p16, p17, p18]                      
        arLT = [ar1, ar2, ar3, ar4, ar5, ar6, ar7, ar8, ar9, ar10, ar11, ar12, ar13, ar14, ar15, ar16, ar17, ar18]
        
        # MinMax for LRW Composition and clutter suppression under changes       
        IdxST = sorted(range(len(pST)), key=lambda sub: pST[sub], reverse=False)[:10]  # sorting
        IdxLT = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=False)[:10]  # sorting
       
        TestVec =Itest.ravel()
        
        CDmcAvg = []; CDmcMin=[]
        for i in range(len(TestVec)):
            ST1 = arST[IdxST[0]][i]; ST2=arST[IdxST[1]][i]; ST3=arST[IdxST[2]][i]; ST4=arST[IdxST[3]][i]
            ST5=arST[IdxST[4]][i]
               
                
            LT1 = arLT[IdxLT[0]][i]; LT2=arLT[IdxLT[1]][i]; LT3=arLT[IdxLT[2]][i]; LT4=arLT[IdxLT[3]][i]
            STr = ST1+ST2; LTr =  (LT2+LT3)/1
            res = STr - LTr 
            CDmcAvg.append(res)
            
    # Robust Long-term (with Interference Mitigation)   
    elif intratio <= 0.9 and test_type == 'MSARkh':
        
        Itest = Data[0]
        
        IrefA = Data[1]; IrefB = Data[2]; IrefC = Data[3]; IrefD = Data[4]; IrefE = Data[5]; IrefF = Data[6]
        IrefG = Data[7]; IrefH = Data[8]; IrefI = Data[9]; IrefJ = Data[10]; IrefK = Data[11]; IrefL = Data[12]
        IrefM = Data[13]; IrefN = Data[14]; IrefO = Data[15]; IrefP = Data[16]; IrefQ = Data[17]; IrefR = Data[18]
        IrefS = Data[19]; IrefT = Data[20]; IrefU = Data[21]; IrefV = Data[22]; IrefX = Data[23];   

        tp = Data[24]; N = Data[25]; K = Data[26]; s = Data[27]
        test_type = Data[28]

        
        
        
        ar19 = test(Itest, IrefS, N,K)
        ar20 = test(Itest, IrefT, N,K) 
        ar21 = test(Itest, IrefU, N,K)
        ar22 = test(Itest, IrefV, N,K) 
        ar23 = test(Itest, IrefX, N,K) 
        
        
        
        p19 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar19).flatten())[0][1])
        p20 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar20).flatten())[0][1])
        p21 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar21).flatten())[0][1])
        p22 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar22).flatten())[0][1])
        p23 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar23).flatten())[0][1])
        

        pST =  [ p19,  p20, p21, p22, p23]                      
        arST = [ ar19, ar20, ar21, ar22, ar23] 
        
        
        ar1 = test(Itest, IrefA, N, K)
        ar2 = test(Itest, IrefB, N, K) 
        ar3 = test(Itest, IrefC, N, K) 
        ar4 = test(Itest, IrefD, N, K) 
        ar5 = test(Itest, IrefE, N,K) 
        ar6 = test(Itest, IrefF, N,K) 
        
        p1 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar1).flatten())[0][1])
        p2 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar2).flatten())[0][1])
        p3 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar3).flatten())[0][1])
        p4 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar4).flatten())[0][1])
        p5 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar5).flatten())[0][1])
        p6 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar6).flatten())[0][1])
        
        ar7 = test(Itest, IrefG, N,K) 
        ar8 = test(Itest, IrefH, N,K)
        ar9 = test(Itest, IrefI, N, K) 
        ar10 = test(Itest, IrefJ, N, K) 
        ar11 = test(Itest, IrefK, N, K) 
        
        p7 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar7).flatten())[0][1])
        p8 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar8).flatten())[0][1])
        p9 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar9).flatten())[0][1])
        p10 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar10).flatten())[0][1])
        p11 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar11).flatten())[0][1])
        
        ar12 = test(Itest, IrefL, N, K) 
        ar13 = test(Itest, IrefM, N,K) 
        ar14 = test(Itest, IrefN, N,K) 
        ar15 = test(Itest, IrefO, N,K) 
        ar16 = test(Itest, IrefP, N,K) 
        ar17 = test(Itest, IrefQ, N,K) 
        ar18 = test(Itest, IrefR, N,K)
        
        p12 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar12).flatten())[0][1])
        p13 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar13).flatten())[0][1])
        p14 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar14).flatten())[0][1])
        p15 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar15).flatten())[0][1])
        p16 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar16).flatten())[0][1])
        
        p17 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar17).flatten())[0][1])
        p18 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar18).flatten())[0][1])
        
        pLT = [p1,p2,p3,p4, p5, p6, p7, p8, p9,p10,p11,p12, p13, p14, p15, p16, p17, p18]                      
        arLT = [ar1, ar2, ar3, ar4, ar5, ar6, ar7, ar8, ar9, ar10, ar11, ar12, ar13, ar14, ar15, ar16, ar17, ar18]
        
        # MinMax for LRW Composition and clutter suppression under changes
        IdxST = sorted(range(len(pST)), key=lambda sub: pST[sub], reverse=False)[:10]  # sorting
        IdxLT = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=False)[:10]  # sorting
       
        TestVec =Itest.ravel()
        
        CDmcAvg = []; CDmcMin=[]
        for i in range(len(TestVec)):
            ST1 = arST[IdxST[0]][i]; ST2=arST[IdxST[1]][i]; ST3=arST[IdxST[2]][i]; ST4=arST[IdxST[3]][i]
            ST5=arST[IdxST[4]][i]
               
                
            LT1 = arLT[IdxLT[0]][i]; LT2=arLT[IdxLT[1]][i]; LT3=arLT[IdxLT[2]][i]; LT4=arLT[IdxLT[3]][i]
            STr = ST1+ST2; LTr =  (LT2+LT1)/1
            res = STr - LTr 
            CDmcAvg.append(res)
            
    
    # Long-term without Interference Mitigation       
    elif test_type == 'MSARk': 
        Itest = Data[0]
        
        IrefA = Data[1]; IrefB = Data[2]; IrefC = Data[3]; IrefD = Data[4]; IrefE = Data[5]; IrefF = Data[6]
        IrefG = Data[7]; IrefH = Data[8]; IrefI = Data[9]; IrefJ = Data[10]; IrefK = Data[11]; IrefL = Data[12]
        IrefM = Data[13]; IrefN = Data[14]; IrefO = Data[15]; IrefP = Data[16]; IrefQ = Data[17]; IrefR = Data[18]
        IrefS = Data[19]; IrefT = Data[20]; IrefU = Data[21]; IrefV = Data[22]; IrefX = Data[23];   

        tp = Data[24]; N = Data[25]; K = Data[26]; s = Data[27]
        test_type = Data[28]

        
        
        
        ar19 = test(Itest, IrefS, N,K)
        ar20 = test(Itest, IrefT, N,K) 
        ar21 = test(Itest, IrefU, N,K)
        ar22 = test(Itest, IrefV, N,K) 
        ar23 = test(Itest, IrefX, N,K) 
        
        
        
        p19 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar19).flatten())[0][1])
        p20 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar20).flatten())[0][1])
        p21 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar21).flatten())[0][1])
        p22 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar22).flatten())[0][1])
        p23 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar23).flatten())[0][1])
        

        pST =  [ p19,  p20, p21, p22, p23]                      
        arST = [ ar19, ar20, ar21, ar22, ar23] 
        
        
        ar1 = test(Itest, IrefA, N, K)
        ar2 = test(Itest, IrefB, N, K) 
        ar3 = test(Itest, IrefC, N, K) 
        ar4 = test(Itest, IrefD, N, K) 
        ar5 = test(Itest, IrefE, N,K) 
        ar6 = test(Itest, IrefF, N,K) 
        
        p1 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar1).flatten())[0][1])
        p2 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar2).flatten())[0][1])
        p3 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar3).flatten())[0][1])
        p4 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar4).flatten())[0][1])
        p5 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar5).flatten())[0][1])
        p6 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar6).flatten())[0][1])
        
        ar7 = test(Itest, IrefG, N,K) 
        ar8 = test(Itest, IrefH, N,K)
        ar9 = test(Itest, IrefI, N, K) 
        ar10 = test(Itest, IrefJ, N, K) 
        ar11 = test(Itest, IrefK, N, K) 
        
        p7 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar7).flatten())[0][1])
        p8 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar8).flatten())[0][1])
        p9 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar9).flatten())[0][1])
        p10 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar10).flatten())[0][1])
        p11 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar11).flatten())[0][1])
        
        ar12 = test(Itest, IrefL, N, K) 
        ar13 = test(Itest, IrefM, N,K) 
        ar14 = test(Itest, IrefN, N,K) 
        ar15 = test(Itest, IrefO, N,K) 
        ar16 = test(Itest, IrefP, N,K) 
        ar17 = test(Itest, IrefQ, N,K) 
        ar18 = test(Itest, IrefR, N,K)
        
        p12 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar12).flatten())[0][1])
        p13 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar13).flatten())[0][1])
        p14 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar14).flatten())[0][1])
        p15 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar15).flatten())[0][1])
        p16 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar16).flatten())[0][1])
        
        p17 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar17).flatten())[0][1])
        p18 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar18).flatten())[0][1])
        
        pLT = [p1,p2,p3,p4, p5, p6, p7, p8, p9,p10,p11,p12, p13, p14, p15, p16, p17, p18]                      
        arLT = [ar1, ar2, ar3, ar4, ar5, ar6, ar7, ar8, ar9, ar10, ar11, ar12, ar13, ar14, ar15, ar16, ar17, ar18]
        
        # MinMax for LRW Composition and clutter suppression under changes
        IdxST = sorted(range(len(pST)), key=lambda sub: pST[sub], reverse=False)[:10]  # sorting
        IdxLT = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=False)[:10]  # sorting
       
        TestVec =Itest.ravel()
        
        CDmcAvg = []; CDmcMin=[]
        for i in range(len(TestVec)):
            ST1 = arST[IdxST[0]][i]; ST2=arST[IdxST[1]][i]; ST3=arST[IdxST[2]][i]; ST4=arST[IdxST[3]][i]
            ST5=arST[IdxST[4]][i]
               
                
            LT1 = arLT[IdxLT[0]][i]; LT2=arLT[IdxLT[1]][i]; LT3=arLT[IdxLT[2]][i]; LT4=arLT[IdxLT[3]][i]
            STr = ST1+ST2; LTr =  (LT2+LT1)/1
            res = STr - LTr 
            CDmcAvg.append(res)
                
                
    # Short-term (without Interference Mitigation)
    elif test_type == 'MSAR':  
        
        Itest = Data[0]   
        TestVec =Itest.ravel()
        
        ar1 = test(Itest, IrefA, N, K)
        ar2 = test(Itest, IrefB, N, K) 
        ar3 = test(Itest, IrefC, N, K) 
        ar4 = test(Itest, IrefD, N, K) 
        ar5 = test(Itest, IrefE, N,K) 
        ar6 = test(Itest, IrefF, N,K) 
        
        p1 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar1).flatten())[0][1])
        p2 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar2).flatten())[0][1])
        p3 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar3).flatten())[0][1])
        p4 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar4).flatten())[0][1])
        p5 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar5).flatten())[0][1])
        p6 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar6).flatten())[0][1])
        
        pLT = [p1,p2,p3,p4, p5, p6]                      
        arLT = [ar1, ar2, ar3, ar4, ar5, ar6]
        
        IdxST = sorted(range(len(pST)), key=lambda sub: pST[sub], reverse=False)[:2]  # sorting
        IdxLT = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=False)[:2]  # sorting
        criteria = ['avg', 'min']
        criteria = criteria[0]
        
        
        
        
        CDmcAvg = []; CDmcMin=[]
        for i in range(len(TestVec)):
            if criteria == 'avg':
                ST1 = arST[IdxST[0]][i]; ST2=arST[IdxST[1]][i]
                LT1 = arLT[IdxLT[0]][i]; LT2=arLT[IdxLT[1]][i]
                
                res = ST1 - LT1   # change map from clutter reduction
                
            elif criteria == 'min':
                res =arST[IdxST[0]][i]  # change map

            CDmcAvg.append(res)

        ICD3 = np.reshape(CDmcAvg, (N,N))
        
    if intratio > 0.9 and test_type == 'MSARh':
        Itest = sST[IdxST[2]]
        
        IrefA = Data[1]; IrefB = Data[2]; IrefC = Data[3]; IrefD = Data[4]; IrefE = Data[5]; IrefF = Data[6]
        
        IrefS = Data[19]; IrefT = Data[20]; IrefU = Data[21]; IrefV = Data[22]; IrefX = Data[23];   

        tp = Data[24]; N = Data[25]; K = Data[26]; s = Data[27]
        test_type = Data[28]

        
        
        
        ar19 = test(Itest, IrefS, N,K)
        ar20 = test(Itest, IrefT, N,K) 
        ar21 = test(Itest, IrefU, N,K)
        ar22 = test(Itest, IrefV, N,K) 
        ar23 = test(Itest, IrefX, N,K) 
        
        
        
        p19 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar19).flatten())[0][1])
        p20 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar20).flatten())[0][1])
        p21 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar21).flatten())[0][1])
        p22 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar22).flatten())[0][1])
        p23 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar23).flatten())[0][1])
        

        pST =  [ p19,  p20, p21, p22, p23]                      
        arST = [ ar19, ar20, ar21, ar22, ar23] 
        
        
        ar1 = test(Itest, IrefA, N, K)
        ar2 = test(Itest, IrefB, N, K) 
        ar3 = test(Itest, IrefC, N, K) 
        ar4 = test(Itest, IrefD, N, K) 
        ar5 = test(Itest, IrefE, N,K) 
        ar6 = test(Itest, IrefF, N,K) 
        
        p1 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar1).flatten())[0][1])
        p2 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar2).flatten())[0][1])
        p3 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar3).flatten())[0][1])
        p4 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar4).flatten())[0][1])
        p5 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar5).flatten())[0][1])
        p6 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar6).flatten())[0][1])
        
        
        pLT = [p1,p2,p3,p4, p5, p6]                      
        arLT = [ar1, ar2, ar3, ar4, ar5, ar6]
        
        # MinMax for LRW Composition and clutter suppression under changes       
        IdxST = sorted(range(len(pST)), key=lambda sub: pST[sub], reverse=False)[:4]  # sorting
        IdxLT = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=False)[:4]  # sorting
       
        TestVec =Itest.ravel()
        
        CDmcAvg = []; CDmcMin=[]
        for i in range(len(TestVec)):
            ST1 = arST[IdxST[0]][i]; ST2=arST[IdxST[1]][i]; ST3=arST[IdxST[2]][i]; ST4=arST[IdxST[3]][i]
            
               
                
            LT1 = arLT[IdxLT[0]][i]; LT2=arLT[IdxLT[1]][i]; LT3=arLT[IdxLT[2]][i]; LT4=arLT[IdxLT[3]][i]
            STr = ST1+ST2; LTr =  (LT2+LT3)/1
            res = STr - LTr 
            CDmcAvg.append(res)
            
    # Robust Long-term (with Interference Mitigation)   
    elif intratio <= 0.9 and test_type == 'MSARh':
        
        Itest = Data[0]
        
        IrefA = Data[1]; IrefB = Data[2]; IrefC = Data[3]; IrefD = Data[4]; IrefE = Data[5]; IrefF = Data[6]
        
        IrefS = Data[19]; IrefT = Data[20]; IrefU = Data[21]; IrefV = Data[22]; IrefX = Data[23];   

        tp = Data[24]; N = Data[25]; K = Data[26]; s = Data[27]
        test_type = Data[28]

        
        
        
        ar19 = test(Itest, IrefS, N,K)
        ar20 = test(Itest, IrefT, N,K) 
        ar21 = test(Itest, IrefU, N,K)
        ar22 = test(Itest, IrefV, N,K) 
        ar23 = test(Itest, IrefX, N,K) 
        
        
        
        p19 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar19).flatten())[0][1])
        p20 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar20).flatten())[0][1])
        p21 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar21).flatten())[0][1])
        p22 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar22).flatten())[0][1])
        p23 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar23).flatten())[0][1])
        

        pST =  [ p19,  p20, p21, p22, p23]                      
        arST = [ ar19, ar20, ar21, ar22, ar23] 
        
        
        ar1 = test(Itest, IrefA, N, K)
        ar2 = test(Itest, IrefB, N, K) 
        ar3 = test(Itest, IrefC, N, K) 
        ar4 = test(Itest, IrefD, N, K) 
        ar5 = test(Itest, IrefE, N,K) 
        ar6 = test(Itest, IrefF, N,K) 
        
        p1 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar1).flatten())[0][1])
        p2 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar2).flatten())[0][1])
        p3 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar3).flatten())[0][1])
        p4 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar4).flatten())[0][1])
        p5 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar5).flatten())[0][1])
        p6 = abs(np.corrcoef(np.array(Itest).flatten(),np.array(ar6).flatten())[0][1])
        
        
        pLT = [p1,p2,p3,p4, p5, p6]                      
        arLT = [ar1, ar2, ar3, ar4, ar5, ar6]
        
        # MinMax for LRW Composition and clutter suppression under changes
        IdxST = sorted(range(len(pST)), key=lambda sub: pST[sub], reverse=False)[:4]  # sorting
        IdxLT = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=False)[:4]  # sorting
       
        TestVec =Itest.ravel()
        
        CDmcAvg = []; CDmcMin=[]
        for i in range(len(TestVec)):
            ST1 = arST[IdxST[0]][i]; ST2=arST[IdxST[1]][i]; ST3=arST[IdxST[2]][i]; ST4=arST[IdxST[3]][i]
            
               
                
            LT1 = arLT[IdxLT[0]][i]; LT2=arLT[IdxLT[1]][i]; LT3=arLT[IdxLT[2]][i]; LT4=arLT[IdxLT[3]][i]
            STr = ST1+ST2; LTr =  (LT2+LT1)/1
            res = STr - LTr 
            CDmcAvg.append(res)
            
        
        
    
    #------------------------- Anomaly Detection ------------------------#

    ICD3 = np.reshape(CDmcAvg, (N,N))
    resARAvg3=moving_average(ICD3, K)
    resARAvgVecC3= resARAvg3.ravel()
    
    resARAvgVecC3= resARAvgVecC3.ravel()
      
   
    
    ## IQR / Boxplot
    percentiles= np.array([75])
    x_p = np.percentile(resARAvgVecC3, percentiles)
    y_p = percentiles/100.0
    
    quartile_1, quartile_3 = np.percentile(resARAvgVecC3, [25, 75]) #25, 75
    
    iqr = quartile_3 - quartile_1
    
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
      
    arraylist = []; Res = []; cnt=0; ths = 1
    for i in range(len(resARAvgVecC3)):
        if (resARAvgVecC3[i] >= lower_bound and resARAvgVecC3[i] <= upper_bound):
            res = 0
            Res.append(res)
       
        else:
            res=resARAvgVecC3[i] 
            Res.append(res)
            
        arraylist.append(Res)

    ICD=np.reshape(arraylist[0], (N, N))
    
 
    return ICD  #,  ICD3, upper_bound
  
  
   

   