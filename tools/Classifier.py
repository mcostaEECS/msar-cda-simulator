from __future__ import division

# NumPy & SciPy
import numpy as np
from numpy import zeros, sqrt, mean, linspace, concatenate, cumsum
from scipy.stats import norm
from scipy.spatial.distance import cdist

# Image Processing
import cv2
from PIL import Image
from skimage.morphology import disk
from skimage.filters.rank import median

# Plotting
import matplotlib.pyplot as plt

# Data Handling
import scipy.io as sio

# Custom module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.load_data import load_data


def Classifier(ICD,tp,pfa, TestType):

   
    th=pfa
    radius = 12
    Imc=np.array(ICD)*10

    ImAMax = Imc.max(axis=0).max(axis=0)
    (thresh, im_bw) = cv2.threshold(Imc, th, ImAMax, cv2.THRESH_BINARY)
    im_bwN = im_bw
    
        
    #OPERACOES MORFOLOGICAS:

    kernel1= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel2= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
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

    return detected_targets, falseAlarm, mission_targets

