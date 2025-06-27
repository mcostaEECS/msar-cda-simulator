from __future__ import division
import numpy as np
import time
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io
import cv2
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from load_data import load_data
import pyprind
import re
import scipy.io as sio
from Classifier import Classifier
import datetime
import rasterio
import numpy as np
from matplotlib.patches import Rectangle
import rasterio.plot
from rasterio.plot import show
import matplotlib.pyplot as plt
from rasterio.plot import show_hist
from osgeo import gdal
import numpy as np
import time
from numpy import zeros,sqrt, mean,linspace,concatenate, cumsum
from scipy.stats import norm
import cv2
from matplotlib import pyplot as plt
import scipy.io as sio
from PIL import Image
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import median
import cv2
from load_data import load_data
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from matplotlib.widgets  import RectangleSelector
import matplotlib.patches as patches
import scienceplots
from matplotlib.collections import LineCollection

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

def Classifier(ICD,tp,pfa, TestType):

    radius = 12

    Imc=np.array(ICD)

    if TestType == 'GLRT':

        ImAMax = Imc.max(axis=0).max(axis=0)
        th=pfa
        (thresh, im_bw) = cv2.threshold(Imc, th, ImAMax, cv2.THRESH_BINARY)
        im_bwN = cv2.normalize(im_bw, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    elif TestType == 'LMP':
        ImAMax = Imc.max(axis=0).max(axis=0)
        th=pfa
        (thresh, im_bw) = cv2.threshold(Imc, th, ImAMax, cv2.THRESH_BINARY)

        im_bwN = im_bw


    #OPERACOES MORFOLOGICAS:
    kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    erosion = cv2.erode(im_bwN,kernel,iterations = 1)
    dilation1 = cv2.dilate(erosion,kernel,iterations = 1)
    dilation2 = cv2.dilate(dilation1,kernel,iterations = 1)
    Imc=dilation2

    fig = plt.figure('test')
    plt.suptitle('Binarizada')
    
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(Imc, cmap = plt.cm.gray)
    plt.axis("off")
    
    plt.show() 
    
    results= cv2.connectedComponentsWithStats(np.array(Imc, dtype=np.uint8),8)

    num_Objects= results[0]
    centroids = results[3]
    centro = centroids[np.argsort(centroids[:, 0])]

    
    if num_Objects > 0:

        outlines = []
        detected = [] 
        for k in range(len(tp)):
            for m in range(len(centro)):
                if (np.abs(centro[m][0] - tp[k][0]) < radius) and (np.abs(centro[m][1] - tp[k][1]) < radius):
                    detected.append(m)
                else:
                    outlines.append(m)
        
        detected_dict = {i:detected.count(i) for i in detected}
        outlines_dict = {i:outlines.count(i) for i in outlines}
        detected_idx = list(detected_dict.keys())
        outlines_idx = list(outlines_dict.keys())

        outlines_idx = [ x for x in outlines_idx if not x in detected_idx]

        potential_targets = num_Objects 
        detected_targets= len(detected_idx) # ok
        if detected_targets > len(tp):
            detected_targets = len(tp)
            falseAlarm = np.abs(potential_targets - len(tp))
        else:
            falseAlarm = np.abs(potential_targets - detected_targets)
    else:
        detected_targets = 0
        falseAlarm = 0
        
    mission_targets = len(tp)    

 
    f, ax = plt.subplots()
    
    for k in outlines_idx:
        #print(k)
        circleOutlines = plt.Circle((centro[k][0], centro[k][1]), 15, color='r', fill=False)
        ax.add_patch(circleOutlines)
        ax.imshow(Imc, cmap='gray', interpolation='nearest')
    for j in detected_idx:
        circleDetected = plt.Circle((centro[j][0], centro[j][1]), 15, color='g', fill=False)
        ax.imshow(Imc, cmap='gray', interpolation='nearest')
        ax.add_patch(circleDetected)

    plt.show()
    
    
    return detected_targets, falseAlarm, mission_targets


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



def test2(Itest, Iref):
    
    TestVec =Itest.ravel()
    RefVec = Iref.ravel()
  
    rho = np.corrcoef(TestVec,RefVec)[0][1]
    unrho = np.sqrt(1-rho**2)
 
    v =4
    CD0 = []
    for i in range(len(TestVec)):
        
        resTarget = TestVec[i]*rho + RefVec[i]*unrho
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
    
    return resARAvg3.ravel()


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
    
    return resARAvg3.ravel()





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
                res = (TestVec[i] + ST2 +ST1 )/3  # change map from clutter reduction
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

    T = 1  # 0 = glrt / 1 = lmp
    TestType = ['MCnAR1', 'LMP']
    block = [100, 250]
    test_type = ['ST', 'MT', 'LT1','LT2']
    block= block[1]
    


    
    # print(g)
    #idx =0
    h = ['free', 'noise', 'correct']
   
    
    test_type = 'ST1'
    test_type = 'LT2'
    #test_type = 'TrainData'
    N = 100; K=9
    
    
    
    
    
    RESAvgCT=[]; RESAvgCC = []; upper_bound=[]
    
    for idx in range(len(h)):
        
        
        if test_type == 'LT2':
            if h[idx] == 'free':
                z = 0
                tag = 1
                
                s= 14
                par=load_data(TestType[0])[s]
                
                start1=2150; end1 = 2250; start2=1300; end2 = 1400  # F1  (100 C-T)
                resAvgCT = MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag)
        
                start1=2150; end1 = 2250; start2=1500; end2 = 1600  # F1  (100 C-C)
                resAvgCC = MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag)
                
                percentiles= np.array([75])
                x_p = np.percentile(resAvgCT, percentiles)
                y_p = percentiles/100.0
                
                quartile_1, quartile_3 = np.percentile(resAvgCT, [25, 75])
                iqr = quartile_3 - quartile_1
                
                
                upper_bound250 = quartile_3 + (iqr * 1.5)
                
                
                
            elif h[idx] == 'noise':
                z = 0
                tag = 0
                
                s= 15
                par=load_data(TestType[0])[s]
                
                start1=2150; end1 = 2250; start2=1300; end2 = 1400  # F1  (100 C-T)
                resAvgCT = MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag)
        
                start1=2150; end1 = 2250; start2=1500; end2 = 1600  # F1  (100 C-C)
                resAvgCC = MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag)
                
                percentiles= np.array([75])
                x_p = np.percentile(resAvgCT, percentiles)
                y_p = percentiles/100.0
                
                quartile_1, quartile_3 = np.percentile(resAvgCT, [25, 75])
                iqr = quartile_3 - quartile_1
                
                
                upper_bound250 = quartile_3 + (iqr * 1.5)
                    
            elif h[idx] == 'correct':
                z = 1
                tag = 0
                
                s= 15
                par=load_data(TestType[0])[s]
                
                start1=2150; end1 = 2250; start2=1300; end2 = 1400  # F1  (100 C-T)
                resAvgCT = MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag)
        
                start1=2150; end1 = 2250; start2=1500; end2 = 1600  # F1  (100 C-C)
                resAvgCC = MCnAR1LT2(start1,start2, end1, end2, N,K,z, tag)
                
                percentiles= np.array([75])
                x_p = np.percentile(resAvgCT, percentiles)
                y_p = percentiles/100.0
                
                quartile_1, quartile_3 = np.percentile(resAvgCT, [25, 75])
                iqr = quartile_3 - quartile_1
                
                
                upper_bound250 = quartile_3 + (iqr * 1.5)
                    
        ##########           
            
        elif test_type == 'ST1': 
            if h[idx] == 'free':
                    
                z = 0
                tag=1
                
                s= 14
                par=load_data(TestType[0])[s]
            
                start1=2150; end1 = 2250; start2=1300; end2 = 1400  # F1  (100 C-T)
                resAvgCT  = MCnAR1ST1(start1,start2, end1, end2, N,K, z, tag)

                start1=2150; end1 = 2250; start2=1500; end2 = 1600  # F1  (100 C-C)
                resAvgCC = MCnAR1ST1(start1,start2, end1, end2, N,K, z, tag)
                
                percentiles= np.array([75])
                x_p = np.percentile(resAvgCT, percentiles)
                y_p = percentiles/100.0
                
                quartile_1, quartile_3 = np.percentile(resAvgCT, [25, 75])
                iqr = quartile_3 - quartile_1
                
                
                upper_bound250 = quartile_3 + (iqr * 1.5)
            
            elif h[idx] == 'noise':
                
                z = 0
                tag = 0
                
                
                
                s= 15
                par=load_data(TestType[0])[s]
                
                start1=2150; end1 = 2250; start2=1300; end2 = 1400  # F1  (100 C-T)
                resAvgCT  = MCnAR1ST1(start1,start2, end1, end2, N,K, z, tag)

                start1=2150; end1 = 2250; start2=1500; end2 = 1600  # F1  (100 C-C)
                resAvgCC = MCnAR1ST1(start1,start2, end1, end2, N,K, z, tag)
                
                percentiles= np.array([75])
                x_p = np.percentile(resAvgCT, percentiles)
                y_p = percentiles/100.0
                
                quartile_1, quartile_3 = np.percentile(resAvgCT, [25, 75])
                iqr = quartile_3 - quartile_1
                
                
                upper_bound250 = quartile_3 + (iqr * 1.5)
                    
                    
            elif h[idx] == 'correct':
            
                z = 1
                tag = 0
                
                
                
                s= 15
                par=load_data(TestType[0])[s]
                
                start1=2150; end1 = 2250; start2=1300; end2 = 1400  # F1  (100 C-T)
                resAvgCT  = MCnAR1ST1(start1,start2, end1, end2, N,K, z, tag)

                start1=2150; end1 = 2250; start2=1500; end2 = 1600  # F1  (100 C-C)
                resAvgCC = MCnAR1ST1(start1,start2, end1, end2, N,K, z, tag)
                
                percentiles= np.array([75])
                x_p = np.percentile(resAvgCT, percentiles)
                y_p = percentiles/100.0
                
                quartile_1, quartile_3 = np.percentile(resAvgCT, [25, 75])
                iqr = quartile_3 - quartile_1
                
                
                upper_bound250 = quartile_3 + (iqr * 1.5)

            

                
        elif test_type == 'TrainData':
            #start1=580; end1 = 830; start2=450; end2 = 700 # C-T  250
            start1=650; end1 = 750; start2=550; end2 = 650 # C-T  100
            resAvgCT = MCnAR1D(start1,start2, end1, end2, N,K)
            ItestCT = MCnAR1D(start1,start2, end1, end2, N,K)
            
        
        
        RESAvgCT.append(resAvgCT); RESAvgCC.append(resAvgCC); upper_bound.append(upper_bound250)
            
                #ItestCT = par[25][start1:end1,start2:end2] 
                
                # fig = plt.figure('test')
                # plt.suptitle('REF / TEST / Change')
                
                # ax = fig.add_subplot(1, 1, 1)
                # plt.imshow(resAvgCT, cmap = plt.cm.gray)
                # plt.axis("off")
                

                
            # plt.show() 
            
            # print(g)
    
            # start1=580; end1 = 830; start2=150; end2 = 400 # C-C
            # resAvgCC = MCnAR1ST(start1,start2, end1, end2, N,K)
        
            # start1=350; end1 = 600; start2=450; end2 = 700 # T-C 
            # resAvgTC= MCnAR1ST(start1,start2, end1, end2, N,K)
            
    
    
    
        
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
            
            
            
            sns.kdeplot(np.array(RESAvgCC[0][0:5000]),color='slategray', lw=2,  alpha=0.8,ax=ax2, label=cc[0]) 
            
            sns.kdeplot(np.array(RESAvgCC[1][0:5000]),color='slategray',lw=2, linestyle='dotted', alpha=0.8,ax=ax2, label=cc[2])  
            sns.kdeplot(np.array(RESAvgCC[2][0:5000]),color='slategray', linestyle='dashdot', marker='o', alpha=0.25,ax=ax2, label=cc[1]) 
            
            sns.kdeplot(np.array(RESAvgCT[0][0:5000]), color='rosybrown', lw=2, alpha=0.8,ax=ax2, label=ct[0]) 
            sns.kdeplot(np.array(RESAvgCT[1][0:5000]),color='rosybrown', lw=2, linestyle='dotted', alpha=0.8,ax=ax2, label=ct[2]) 
            sns.kdeplot(np.array(RESAvgCT[2][0:5000]),color='rosybrown', alpha=0.25,  marker='o', linestyle='dashdot',ax=ax2, label=ct[1]) 

            ax2.axvline(upper_bound[0], 0,1, color='red', alpha=0.35, label=r'threshold=%.2f'%upper_bound[0]) 
            
        
           
           
           
            ax2.legend(prop={'size': 20})
            ax2.set_ylabel(r'Density', fontsize=30)  
            ax2.yaxis.set_label_position("right")
            ax2.set_xlabel(r'Standardized value (z-score)', fontsize=30)  
            ax2.xaxis.set_label_coords(.5, -.085)
            ax2.yaxis.set_label_coords(-0.10, 0.5)
            ax2.set_xlim([0, 0.08])
            ax2.yaxis.set_tick_params(labelleft=False, labelright=True)


        
           

            #ax1.legend(loc='lower center', bbox_to_anchor=(1, 0.985), fancybox=True, shadow=True, ncol=4)
            


            # ax1.text(0.5,-0.09, "(a)", size=12, ha="center", 
            #         transform=ax1.transAxes)
            # ax2.text(0.5,-0.09, "(b)", size=12, ha="center", 
            #         transform=ax2.transAxes)
    path='/home/marcello-costa/workspace/'
    namefile = 'PDF_%s_Int_%s.png'%(test_type,h[idx])
 
    fig.savefig(path + namefile, dpi=DPI)
    #fig.tight_layout()
            
          

    plt.show()      

            
        
    
 












