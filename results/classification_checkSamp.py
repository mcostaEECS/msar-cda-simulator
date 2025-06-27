from __future__ import division

# Core libraries
import os
import re
import math
import array
import numpy as np
from ast import literal_eval

# Visualization
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap

# Image processing
import cv2
from skimage.morphology import disk
from skimage.filters.rank import median

# Data handling and computation
import scipy.io as sio
from scipy.special import erfcinv
from scipy.spatial.distance import cdist

# Custom module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.load_sample import load_sample


#------------------------- Morfological Filtering/Alarms computation ------------------------#
def ClassifierLocal(ICD,tp,pfa, TestType, pair, TP, N):

            radius = 12
            Imc=np.array(ICD)*10 # gain due Spatial filtering
            th=pfa
            ImAMax = Imc.max(axis=0).max(axis=0)
            (thresh, im_bw) = cv2.threshold(Imc, th, ImAMax, cv2.THRESH_BINARY)
            im_bwN = im_bw

            #Morphologic operators
            kernel1= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            kernel2= cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            erosion = cv2.erode(im_bwN,kernel1,iterations = 1)
            dilation1 = cv2.dilate(erosion,kernel1,iterations = 1)
            dilation2 = cv2.dilate(dilation1,kernel2,iterations = 1)
            Imc=dilation2


            u8 =Imc.astype(np.uint8)
            nb_components, output, stats, centroids0 = cv2.connectedComponentsWithStats(u8, connectivity=8)
            sizes = stats[1:, -1]

            centroids0 = centroids0[1:]
            centroids=[]
            for i in range(len(centroids0)):
                if sizes[i]>30 and sizes[i]<800:
                    centroids.append(centroids0[i])
            centroids = np.array(centroids)
            num_Objects = len(centroids)
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
                detected_targets= len(detected_idx) 
                if detected_targets > len(tp):
                    detected_targets = len(tp)
                    falseAlarm = np.abs(potential_targets - len(tp))
                else:
                    falseAlarm = np.abs(potential_targets - detected_targets)
            else:
                detected_targets = 0
                falseAlarm = 0

            mission_targets = len(tp)

            # if pair==0 or pair==1 or pair==2 or pair==3 or pair==4 or pair==5:
            #    start=500; end = 900; Start=-500
            # if pair==6 or pair==7 or pair==8 or pair==9 or pair==10 or pair==11:
            #    start=100; end = 700; Start=-100
            # if pair==12 or pair==13 or pair==14 or pair==15 or pair==16 or pair==17:
            #    start=2120; end = 2520 ; Start=-2120
            # if pair==18 or pair==19 or pair==20 or pair==21 or pair==22 or pair==23:
            #    start=2420; end = 2820 ; Start=-2420
               
            value = 5 #cmap='PuBu_r','gist_gray'
            # Get 'gray' colormap and slice from 0.5 to 1.0 (gray to white)
            gray_to_white = plt.get_cmap('viridis')

            new_cmap = gray_to_white(np.linspace(0.55, 1.0, 256))  # Create slice
            custom_cmap = LinearSegmentedColormap.from_list("gray_to_white_slice", new_cmap)

            #my_cmap_r = cmap.reversed()
             #f, ax, ax1 = plt.subplots()
            f, ax1 = plt.subplots()
            for k in outlines_idx:
                grey_new = np.where((17 -Imc) < value,17,Imc+value)
                circleOutlines = plt.Circle((centro[k][0], centro[k][1]), 15, color='r', fill=False)
                ax1.add_patch(circleOutlines)
                ax1.imshow(Imc, cmap = custom_cmap, interpolation='nearest')
            for j in detected_idx:
                circleDetected = plt.Circle((centro[j][0], centro[j][1]), 15, color='k', fill=False)
                ax1.imshow(Imc, cmap = custom_cmap, interpolation='nearest')
                ax1.add_patch(circleDetected)
                
            plt.axis("off") 
            path='results/plots/'
            f.savefig(path + 'ClassSample_%s_%s.png'%(TestType, pair), dpi=350,bbox_inches='tight',transparent=True, pad_inches=0)
            f.tight_layout()
            plt.show()

            return detected_targets, falseAlarm, mission_targets


if __name__ == "__main__":

    test_type = ['MSAR',  'MSARk', 'MSARkh']
    test_type  = test_type[2]
    N = 125
    K = 9
    pair =2
 
    ch =0  # 1. change map 0. classification 
    
    if pair==0 or pair==1 or pair==2 or pair==3 or pair==4 or pair==5:
        tp = 'F1'; TP=tp
    
    
    if test_type == 'MSAR':
        path = 'results/dataMSAR/dataMSAR_sample/'
        pfa=0.65# due \ell=1
        
        
    elif test_type == 'MSARk':
        path = 'results/dataMSARk/dataMSARk_sample/'
        pfa=1.5 # due \ell=2
        
    elif test_type == 'MSARkh':
        path = 'results/dataMSARkh/dataMSARkh_sample/'
        pfa=3 # due \ell=2
   
    name = 'ICD_%s_N_%d_K_%d_tp_%s_pair_%d'%(test_type,N,K,tp, pair)

    ImageReadCD=sio.loadmat(path+name+'.mat')  
    ICD=ImageReadCD['ICD']

    ICD = ICD.reshape(500, 500)
    path2 = 'data/sample/targets/'
    
    
    if tp == 'F1':
        tp=sio.loadmat(path2+'F1.mat'); tp=tp['F1']; tp = np.fliplr(tp)
        
    par=load_sample(test_type)[pair]
    tp=par[23]; tp =  np.fliplr(tp); TP =par[24]
    
    offset_y = 2050

    # # offset horizontal (colunas)
    offset_x = 1100

    # # Supondo que os pontos estão no formato (x, y) ou (coluna, linha):
    tp_adjusted = tp - np.array([offset_x, offset_y])

   

 
    start1=0; end1 = 500; start2=0; end2 = 500
        
    Itest = par[25][start1:end1,start2:end2] 
    
    MinMag = ICD.min(axis=0).min(axis=0)
    MaxMag = ICD.max(axis=0).max(axis=0)
    AvgMag = ICD.mean(axis=0).mean(axis=0)
    print('Magnitude (min, avg, max):',(MinMag, AvgMag, MaxMag))
    
    
    #------------ Change Map after spatial filt./Classification ------------------------#
   
    if ch == 1:
        fig = plt.figure('test')
        plt.suptitle('Change Map')
        
        ax = fig.add_subplot(1,1, 1)
        plt.imshow(Itest, cmap = plt.cm.gray)
        plt.axis("off")
        plt.show() 
        
    elif ch == 0:

        [detected_targets, falseAlarm, mission_targets]=ClassifierLocal(ICD,tp_adjusted,pfa, test_type, pair, TP, N)
        PD= np.sum(detected_targets)/25
        FAR = np.sum(falseAlarm)/(250000/144)
        
        print('Pair:', pair, ' ', 'Test:',test_type )
        print('Detected Target:', detected_targets)
        print('False Alarms:', falseAlarm)
        print('PD:', PD )
        print('FAR:', FAR )
        
    
    
    
    