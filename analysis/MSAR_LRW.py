from __future__ import division

# --- Core Python --- #
import os
import re
import time
import datetime
import random
import array

# --- NumPy & SciPy --- #
import numpy as np
from numpy import zeros, sqrt, mean, linspace, concatenate, cumsum
from scipy.stats import norm
from scipy.spatial.distance import cdist
import scipy.io as sio

# --- Image and Raster Processing --- #
import cv2
from PIL import Image
from skimage.morphology import disk
from skimage.filters.rank import median
import rasterio
from rasterio.plot import show, show_hist
from osgeo import gdal

# --- Plotting & Visualization --- #
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import scienceplots

# --- Progress bar --- #
import pyprind

# Custom module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.load_sample import load_sample


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

#omitting the sample


def randomwalk1D(ItestCT,arLAGS):
    
    # randomwalk1D(ItestCT,arLT)
    x, y = np.mean(ItestCT), 0
   
    # Generate the time points [1, 2, 3, ... , n]
    timepoints = np.arange(len(arLAGS))
    
    positions = [y]

    directions = ["UP", "DOWN"]
    for i in range(0, len(arLAGS)-1 ):
        #print(i)
        

        # Randomly select either UP or DOWN
        step = random.choice(directions)
        
        cmp = np.mean(arLAGS[i])/x
        
        # Move the object up or down
        if cmp < 1.2:
            TAG = 'UP'
            y1 = test2(ItestCT, arLAGS[i], TAG)
            y = np.mean(y1)
        else:
            TAG = 'DOWN'
            y1 = test2(ItestCT, arLAGS[i], TAG)
            y = np.mean(y1)
            
        a = np.cumsum(y)
 
        # Keep track of the positions
        positions.append(a[0])
 
        # Keep track of the positions
        #positions.append(y)

    return timepoints, positions


def test(Itest, Iref):
    
 
     
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



def test2(Itest, Iref, TAG):
    
 
     
    TestVec =Itest.ravel()
    RefVec = Iref.ravel()
    
    
    rho = np.corrcoef(TestVec,RefVec)[0][1]
    unrho = np.sqrt(1-rho**2)
 
    th = 1
    CD0 = []
    for i in range(len(TestVec)):
        
        if TAG ==  "UP":
            resTarget = TestVec[i]*rho + RefVec[i]*unrho # degree of persistence
        else:
            resTarget = TestVec[i]
    
        CD0.append(resTarget)
        
    return CD0


def MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag):
    
    #ItestCT = par[25][start1:end1,start2:end2]  
    ItestCT = MCnAR1D(start1,start2, end1, end2, N,K,z, tag)
        

    IrefA = par[0][start1:end1,start2:end2]
    IrefB = par[1][start1:end1,start2:end2]
    IrefC = par[2][start1:end1,start2:end2]
    IrefD = par[3][start1:end1,start2:end2]
    IrefE = par[4][start1:end1,start2:end2]
    IrefF = par[5][start1:end1,start2:end2]
    
    IrefG = par[6][start1:end1,start2:end2]
    IrefH = par[7][start1:end1,start2:end2]
    IrefI = par[8][start1:end1,start2:end2]
    IrefJ = par[9][start1:end1,start2:end2]
    IrefK = par[10][start1:end1,start2:end2]
    IrefL = par[11][start1:end1,start2:end2]
    
    IrefM = par[12][start1:end1,start2:end2]
    IrefN = par[13][start1:end1,start2:end2]
    IrefO = par[14][start1:end1,start2:end2]
    IrefP = par[15][start1:end1,start2:end2]
    IrefQ = par[16][start1:end1,start2:end2]
    IrefR = par[17][start1:end1,start2:end2]
    
    IrefS = par[18][start1:end1,start2:end2]
    IrefT = par[19][start1:end1,start2:end2]
    IrefU = par[20][start1:end1,start2:end2]
    IrefV = par[21][start1:end1,start2:end2]
    IrefX = par[22][start1:end1,start2:end2]
    
    arLags = [IrefA, IrefB, IrefC, IrefD, IrefE, IrefF, IrefG, IrefH, IrefI, IrefJ, IrefK, 
              IrefL,IrefM, IrefN, IrefO, IrefP, IrefQ, IrefR]
    
    
    
    rw = randomwalk1D(ItestCT,arLags)
    
    
    
    ar1 = test(ItestCT, IrefA);ar2 = test(ItestCT, IrefB);ar3 = test(ItestCT, IrefC)
    ar4 = test(ItestCT, IrefD);ar5 = test(ItestCT, IrefE);ar6 = test(ItestCT, IrefF)
     
    ar7 = test(ItestCT, IrefG);ar8 = test(ItestCT, IrefH);ar9 = test(ItestCT, IrefI)
    ar10 = test(ItestCT, IrefJ);ar11 = test(ItestCT, IrefK);ar12 = test(ItestCT, IrefL) 
    
    ar13 = test(ItestCT, IrefM);ar14 = test(ItestCT, IrefN);ar15 = test(ItestCT, IrefO)
    ar16 = test(ItestCT, IrefP);ar17 = test(ItestCT, IrefQ);ar18 = test(ItestCT, IrefR) 
    
    ar19 = test(ItestCT, IrefS);ar20 = test(ItestCT, IrefT);ar21 = test(ItestCT, IrefU)
    ar22 = test(ItestCT, IrefV);ar23 = test(ItestCT, IrefX) 
    
    
    p1 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar1).flatten())[0][1])
    p2 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar2).flatten())[0][1])
    p3 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar3).flatten())[0][1])
    p4 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar4).flatten())[0][1])
    p5 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar5).flatten())[0][1])
    p6 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar6).flatten())[0][1])
    
    p7 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar7).flatten())[0][1])
    p8 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar8).flatten())[0][1])
    p9 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar9).flatten())[0][1])
    p10 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar10).flatten())[0][1])
    p11 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar11).flatten())[0][1])
    p12 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar12).flatten())[0][1])
    
    p13 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar13).flatten())[0][1])
    p14 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar14).flatten())[0][1])
    p15 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar15).flatten())[0][1])
    p16 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar16).flatten())[0][1])
    p17 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar17).flatten())[0][1])
    p18 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar18).flatten())[0][1])
    
    
    p19 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar19).flatten())[0][1])
    p20 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar20).flatten())[0][1])
    p21 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar21).flatten())[0][1])
    p22 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar22).flatten())[0][1])
    p23 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar23).flatten())[0][1])
    
    
    pST =  [ p19,  p20, p21, p22, p23]                      
    arST = [ ar19, ar20, ar21, ar22, ar23] 
    
    
    pLT = [p1,p2,p3,p4, p5, p6, p7, p8, p9,p10,p11,p12, p13, p14, p15, p16, p17, p18]                      
    arLT = [ar1, ar2, ar3, ar4, ar5, ar6, ar7, ar8, ar9, ar10, ar11, ar12, ar13, ar14, ar15, ar16, ar17, ar18]
      
   
    IdxST = sorted(range(len(pST)), key=lambda sub: pST[sub], reverse=False)[:4]  # sorting
    IdxLT = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=True)[:4]  # sorting
    criteria = ['avg', 'min']
    criteria = criteria[0]
    
    TestVec =ItestCT.ravel()
    
    
    CDmcAvg = []; CDmcMin=[]
    for i in range(len(TestVec)):
        if z == 0:
            ST1 = (arST[IdxST[0]][i] + arST[IdxST[1]][i])/2
            LT1 = (arLT[IdxLT[0]][i] + arLT[IdxLT[1]][i])/2
            
            res = ST1 - LT1 # change map from clutter reduction
            
        elif z == 1:
            ST1 = (arST[IdxST[1]][i] + arST[IdxST[2]][i])/2
            LT1 = (arLT[IdxLT[1]][i] + arLT[IdxLT[2]][i])/2
            
            res = ST1 - LT1 # change map from clutter reduction

        CDmcAvg.append(res)

    ICD3 = np.reshape(CDmcAvg, (N,N))
    resARAvg3=moving_average(ICD3, K)
    
    return ICD3, rw[0], rw[1], ItestCT


def MCnAR1ST1(start1,start2, end1, end2, N,K, z, tag):
    
    #ItestCT = par[25][start1:end1,start2:end2]  
    
    ItestCT = MCnAR1D(start1,start2, end1, end2, N,K, z, tag)
        

    IrefA = par[0][start1:end1,start2:end2]
    IrefB = par[1][start1:end1,start2:end2]
    IrefC = par[2][start1:end1,start2:end2]
    IrefD = par[3][start1:end1,start2:end2]
    IrefE = par[4][start1:end1,start2:end2]
    IrefF = par[5][start1:end1,start2:end2]
    
    IrefS = par[18][start1:end1,start2:end2]
    IrefT = par[19][start1:end1,start2:end2]
    IrefU = par[20][start1:end1,start2:end2]
    IrefV = par[21][start1:end1,start2:end2]
    IrefX = par[22][start1:end1,start2:end2]
    
    arLags = [IrefA, IrefB, IrefC, IrefD, IrefE, IrefF]
    
    
    
    rw = randomwalk1D(ItestCT,arLags)
    
    
    
    ar19 = test(ItestCT, IrefS);ar20 = test(ItestCT, IrefT);ar21 = test(ItestCT, IrefU)
    ar22 = test(ItestCT, IrefV);ar23 = test(ItestCT, IrefX) 
    
    p19 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar19).flatten())[0][1])
    p20 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar20).flatten())[0][1])
    p21 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar21).flatten())[0][1])
    p22 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar22).flatten())[0][1])
    p23 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar23).flatten())[0][1])
    
    
    pST =  [ p19,  p20, p21, p22, p23]                      
    arST = [ ar19, ar20, ar21, ar22, ar23] 
    
    
    
    
   
    
    ar1 = test(ItestCT, IrefA);ar2 = test(ItestCT, IrefB);ar3 = test(ItestCT, IrefC)
    ar4 = test(ItestCT, IrefD);ar5 = test(ItestCT, IrefE);ar6 = test(ItestCT, IrefF)
     
    
    
    p1 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar1).flatten())[0][1])
    p2 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar2).flatten())[0][1])
    p3 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar3).flatten())[0][1])
    p4 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar4).flatten())[0][1])
    p5 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar5).flatten())[0][1])
    p6 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar6).flatten())[0][1])
    

    
    
    
    
    
    pLT = [p1,p2,p3,p4, p5, p6]        
    print(pLT)              
    arLT = [ar1, ar2, ar3, ar4, ar5, ar6]
      
    IdxST = sorted(range(len(pST)), key=lambda sub: pST[sub], reverse=False)[:4]  # sorting
    IdxLT = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=True)[:4]  # sorting
    print(IdxLT)
    criteria = ['avg', 'min']
    criteria = criteria[0]
    
    TestVec =ItestCT.ravel()
    
    
    CDmcAvg = []; CDmcMin=[]
    for i in range(len(TestVec)):
        if z == 0:
            LT1 = (arLT[IdxLT[0]][i])
            ST1 = (arST[IdxST[0]][i])
            
            res = ST1- LT1   # change map from clutter reduction
            
        elif z == 1:
            LT1 = (arLT[IdxLT[2]][i])
            ST1 = (arST[IdxST[2]][i])
            
            res = ST1 - LT1

        CDmcAvg.append(res)

    ICD3 = np.reshape(CDmcAvg, (N,N))
    resARAvg3=moving_average(ICD3, K)
    
    
    return ICD3, rw[0], rw[1], ItestCT
    
    




def MCnAR1D(start1,start2, end1, end2, N,K, z, tag):
    
    ItestCT = par[25][start1:end1,start2:end2]  #36 / 18 limite de memoria
  
    IrefS = par[18][start1:end1,start2:end2]
    IrefT = par[19][start1:end1,start2:end2]
    IrefU = par[20][start1:end1,start2:end2]
    IrefV = par[21][start1:end1,start2:end2]
    IrefX = par[22][start1:end1,start2:end2]
    
    Irefs = [IrefS.ravel(), IrefT.ravel(), IrefU.ravel(), IrefV.ravel(), IrefX.ravel()]

    
    # fig = plt.figure('test')
    # #plt.suptitle('REF / TEST / Change')
    
    # ax = fig.add_subplot(1, 1, 1)
    # plt.imshow(ItestCT, cmap = plt.cm.gray)
    # plt.axis("off")
    # plt.show() 
    # print(g)
    
    
   
    
    ar19 = test(ItestCT, IrefS);ar20 = test(ItestCT, IrefT);ar21 = test(ItestCT, IrefU)
    ar22 = test(ItestCT, IrefV);ar23 = test(ItestCT, IrefX) 
    
    
    
    p19 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar19).flatten())[0][1])
    p20 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar20).flatten())[0][1])
    p21 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar21).flatten())[0][1])
    p22 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar22).flatten())[0][1])
    p23 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar23).flatten())[0][1])
    
    
    pST =  [ p19,  p20, p21, p22, p23]                      
    arST = [ ar19, ar20, ar21, ar22, ar23] 
    
     
   
    IdxST = sorted(range(len(pST)), key=lambda sub: pST[sub], reverse=False)[:4]  # sorting
    
  
    criteria = ['avg', 'min']
    criteria = criteria[0]
    
    TestVec =ItestCT.ravel()
    
    
    CDmcAvg = []; CDmcMin=[]
    for i in range(len(TestVec)):
        if z == 0:
            
            if tag == 1:
                ST1 = Irefs[IdxST[1]][i]; ST2 = Irefs[IdxST[2]][i]
                res = (TestVec[i] + ST1 + ST2)/3   # change map from clutter reduction
            else:
                ST1 = Irefs[IdxST[0]][i]
                
                res = (TestVec[i] + ST1)/2   # change map from clutter reduction
                
        elif z == 1:
            ST1 = Irefs[IdxST[1]][i]; ST2 = Irefs[IdxST[3]][i]
            res = (TestVec[i] + ST1 + ST2)/3 #ST1  # change map

        CDmcAvg.append(res)

    ICD3 = np.reshape(CDmcAvg, (N,N))
    
 
    
    return ICD3.ravel()


if __name__ == "__main__":

       
    test_type = ['CT', 'CC', 'CTi', 'CTh']
    #test_type = ['CT']
    rep = 1; K = 9; step = 25; d = 10; N =100
   
    
    path = '/home/marcello-costa/workspace/kthSim/Output/RW/'

    block = [100, 500]
    #test_type = ['S1', 'K1', 'F1', 'AF1']
    block= block[1]
    TestType = ['ARn3opt']
    
    


    
    # print(g)
    #idx =0
         
            
    pathOut = '/home/marcello-costa/workspace/kthSim/Output/RW/logs/'
    campaign = 'Campaign_rep_%d_lags_%d_div_%d'%(rep, step, d)
    
    REP = []
    Hatt = []; Rwkt=[]; Rwmt=[]; Itestt=[]; Icdt=[]; Tht=[]
    Hatc = []; Rwkc=[]; Rwmc=[]; Itestc=[]; Icdc=[]; Thc=[]
    Hati = []; Rwki=[]; Rwmi=[]; Itesti=[]; Icdi=[]; Thi=[]
    Hath = []; Rwkh=[]; Rwmh=[]; Itesth=[]; Icdh=[]; Thh=[]
   
    for r in range(rep):
        
        hatt = []; rwkt=[]; rwmt=[]; itestt=[];icdt=[]; tht=[]
        hatc = []; rwkc=[]; rwmc=[]; itestc=[];icdc=[]; thc=[]
        hati = []; rwki=[]; rwmi=[]; itesti=[];icdi=[]; thi=[]
        hath = []; rwkh=[]; rwmh=[]; itesth=[];icdh=[]; thh=[]
        for i in range(len(test_type)):
            print('rep_%d_lags_%d_div_%d_test_%s'%(r,step,d,test_type[i]))
            if test_type[i] == 'CT':
                
                z = 0
                tag = 1
                
                s= 14
                par=load_data(TestType[0])[s]
                
                start1=2150; end1 = 2250; start2=1300; end2 = 1400  # F1  (100 C-T)
                [resAvg,rw1,rw2, Test]  = MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag)
                
                cd =  resAvg.ravel()      
                percentiles= np.array([75])
                x_p = np.percentile(cd, percentiles)
                y_p = percentiles/100.0
                
                quartile_1, quartile_3 = np.percentile(cd, [50, 85])
                iqr = quartile_3 - quartile_1
                upper_bound = quartile_3 + (iqr * 1.5)
                
                CD = []
                for j in range(len(cd)):
                    if cd[j] >= upper_bound:
                        res = cd[j]
                    else:
                        res = 0
                    CD.append(res)
                ICD3 = np.reshape(CD, (N,N))
                
                hatt.append(resAvg); rwkt.append(rw1); rwmt.append(rw2)
                itestt.append(Test); icdt.append(ICD3); tht.append(upper_bound)
                
            elif test_type[i] == 'CC':
                
                z = 0
                tag = 1
                
                s= 14
                par=load_data(TestType[0])[s]
                
                start1=2150; end1 = 2250; start2=1500; end2 = 1600  # F1  (100 C-C)
                [resAvg,rw1,rw2, Test]  = MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag)
                
                cd =  resAvg.ravel()      
                percentiles= np.array([75])
                x_p = np.percentile(cd, percentiles)
                y_p = percentiles/100.0
                
                quartile_1, quartile_3 = np.percentile(cd, [50, 85])
                iqr = quartile_3 - quartile_1
                upper_bound = quartile_3 + (iqr * 1.5)
                
                CD = []
                for j in range(len(cd)):
                    if cd[j] >= upper_bound:
                        res = cd[j]
                    else:
                        res = 0
                    CD.append(res)
                ICD3 = np.reshape(CD, (N,N))
                
                hatc.append(resAvg); rwkc.append(rw1); rwmc.append(rw2)
                itestc.append(Test); icdc.append(ICD3); thc.append(upper_bound)
                
        
            elif test_type[i] == 'CTi':
                z = 0
                tag = 0
                
                s= 15
                par=load_data(TestType[0])[s]
                
                start1=2150; end1 = 2250; start2=1300; end2 = 1400  # F1  (100 C-T)
                [resAvg,rw1,rw2, Test] = MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag)
                
                
                
                cd =  resAvg.ravel()      
                percentiles= np.array([75])
                x_p = np.percentile(cd, percentiles)
                y_p = percentiles/100.0
                
                quartile_1, quartile_3 = np.percentile(cd, [50, 85])
                iqr = quartile_3 - quartile_1
                upper_bound = quartile_3 + (iqr * 1.5)
                
                CD = []
                for j in range(len(cd)):
                    if cd[j] >= upper_bound:
                        res = cd[j]
                    else:
                        res = 0
                    CD.append(res)
                ICD3 = np.reshape(CD, (N,N))
                
                hati.append(resAvg); rwki.append(rw1); rwmi.append(rw2)
                itesti.append(Test); icdi.append(ICD3); thi.append(upper_bound)
                
            elif test_type[i] == 'CTh':
                
                z = 1
                tag = 0
                
                s= 15
                par=load_data(TestType[0])[s]
                
                start1=2150; end1 = 2250; start2=1300; end2 = 1400  # F1  (100 C-T)
                [resAvg,rw1,rw2, Test] = MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag)
                
                cd =  resAvg.ravel()      
                percentiles= np.array([75])
                x_p = np.percentile(cd, percentiles)
                y_p = percentiles/100.0
                
                quartile_1, quartile_3 = np.percentile(cd, [50, 85])
                iqr = quartile_3 - quartile_1
                upper_bound = quartile_3 + (iqr * 1.5)
                
                CD = []
                for j in range(len(cd)):
                    if cd[j] >= upper_bound:
                        res = cd[j]
                    else:
                        res = 0
                    CD.append(res)
                ICD3 = np.reshape(CD, (N,N))
                
                hath.append(resAvg); rwkh.append(rw1); rwmh.append(rw2)
                itesth.append(Test); icdh.append(ICD3); thh.append(upper_bound)

 
        Hatt.append(hatt); Rwkt.append(rwkt); Rwmt.append(rwmt)
        Itestt.append(itestt); Icdt.append(icdt); Tht.append(tht)
        Hatc.append(hatc); Rwkc.append(rwkc); Rwmc.append(rwmc)
        Itestc.append(itestc); Icdc.append(icdc); Thc.append(thc)
        
        Hati.append(hati); Rwki.append(rwki); Rwmi.append(rwmi)
        Itesti.append(itesti); Icdi.append(icdi); Thi.append(thi)
        
        Hath.append(hath); Rwkh.append(rwkh); Rwmh.append(rwmh)
        Itesth.append(itesth); Icdh.append(icdh); Thh.append(thh)
        
        REP.append(r)
        
              
    
    
    results={}
    results['REP.mat']=REP
    results['Hatt.mat']=Hatt
    results['Rwkt.mat']=Rwkt
    results['Rwmt.mat']=Rwmt
    results['Itestt.mat']=Itestt
    results['Icdt.mat']=Icdt
    results['Tht.mat']=Tht
    
    results['Hatc.mat']=Hatc
    results['Rwkc.mat']=Rwkc
    results['Rwmc.mat']=Rwmc
    results['Itestc.mat']=Itestc
    results['Icdc.mat']=Icdc
    results['Thc.mat']=Thc
    
    results['Hati.mat']=Hati
    results['Rwki.mat']=Rwki
    results['Rwmi.mat']=Rwmi
    results['Itesti.mat']=Itesti
    results['Icdi.mat']=Icdi
    results['Thi.mat']=Thi
    
    results['Hath.mat']=Hath
    results['Rwkh.mat']=Rwkh
    results['Rwmh.mat']=Rwmh
    results['Itesth.mat']=Itesth
    results['Icdh.mat']=Icdh
    results['Thh.mat']=Thh
    
    
    id = pathOut+campaign+'.mat'        
    scipy.io.savemat(id,results) 
                
    
    