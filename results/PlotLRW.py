# --- Core Python --- #
import os
import re
import time
import array
import datetime
from ast import literal_eval
import statistics
import random

# --- Scientific and Numerical --- #
import numpy as np
from numpy import zeros, sqrt, mean, linspace, concatenate, cumsum
import scipy.io as sio
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import norm
from scipy.spatial.distance import cdist

# --- Image Processing --- #
import cv2
from PIL import Image
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import median

# --- Raster and Geospatial --- #
import rasterio
from rasterio.plot import show, show_hist
from osgeo import gdal
import gstools as gs

# --- Plotting --- #
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import scienceplots

# --- Progress and Debugging --- #
import pyprind

# Custom module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.load_data import load_data



#------------------------- Split Image ------------------------#

def OrderData(theta, tput_meas):
    Theta = sorted(set(theta))
    TPUT_meas = []
    for k in range(len(Theta)):
        indexes=[i for i, x in enumerate(theta) if x == Theta[k]]
        Tput_meas= 0
        for i in indexes:
            Tput_meas += tput_meas[i]
        TPUT_meas.append(Tput_meas)
    return Theta, TPUT_meas



if __name__ == "__main__":
    
    # repeat 100 x

    K = 9
    step = 23
 
    
    path = 'results/logs/'
    
    files_plots = [i for i in os.listdir(path) if i.endswith('.mat')]
    
    

    log = sio.loadmat(path+files_plots[0])
    REP=log['REP.mat'][0].tolist()
    RWkt=log['Rwkt.mat']
    RWmt=log['Rwmt.mat']
    
    RWkc=log['Rwkc.mat']
    RWmc=log['Rwmc.mat']
    
    RWki=log['Rwki.mat']
    RWmi=log['Rwmi.mat']
    
    RWkh=log['Rwkh.mat']
    RWmh=log['Rwmh.mat']
    

    rw1t=[]; rw2t = []
    rw1c=[]; rw2c = []
    rw1i=[]; rw2i = []
    rw1h=[]; rw2h = []
    for k in range(len(REP)):
        
 
        rw1t.append(RWkt[k][0])
        rw2t.append(RWmt[k][0])
        
        rw1c.append(RWkc[k][0])
        rw2c.append(RWmc[k][0])
        
        rw1i.append(RWki[k][0])
        rw2i.append(RWmi[k][0])
        
        rw1h.append(RWkh[k][0])
        rw2h.append(RWmh[k][0])
        
        
    RW1t = sum(rw1t)[1:]/len(rw1t)
    RW2t = sum(rw2t)[1:]/len(rw2t)
    
    RW1c = sum(rw1c)[1:]/len(rw1c)
    RW2c = sum(rw2c)[1:]/len(rw2c)
    
    RW1i = sum(rw1i)[1:]/len(rw1i)
    RW2i = sum(rw2i)[1:]/len(rw2i)
    
    RW1h = sum(rw1h)[1:]/len(rw1h)
    RW2h = sum(rw2h)[1:]/len(rw2h)
    
      
    
    


#  # # #------------------------ Graphic Analysis -------------------------#
 
    DPI = 250
    plt.rc('text', usetex=True)
    #DPI =200
    
    test_type = 'RW' 
    block = 200


    SMALL_SIZE = 10
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 12

    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    

    
 
    with plt.style.context(['science', 'ieee', 'std-colors']):
            
            fig, ax1 = plt.subplots(1, 1, figsize=(10,3), dpi=DPI) #, constrained_layout=True)
            plt.subplots_adjust(hspace=0.80)
           
            offset = 0
            
            ax1.plot(RW1h+offset, RW2h, color='b', linewidth=2,  alpha=0.35, label=r'$C_{\textnormal{int}}^{h}\to T_{\textnormal{int}}^{h}$') 
            # Find valleys(min).
            valley_indexes = signal.argrelextrema(RW2h, np.less)
            valley_indexes = valley_indexes[0]
            # Plot valleys.
            valley_x = valley_indexes
            valley_y = np.array(RW2h)[valley_indexes]
            #ax1.plot(valley_x, valley_y, color='rosybrown')
       
            
            aa = RW2h.tolist()
            bb = RW1h.tolist()
            
            pointsMin = sorted(range(len(valley_y)), key=lambda sub: valley_y[sub], reverse=False)[:2] 
            indice1 = aa.index(valley_y[pointsMin[0]])
            indice2 = aa.index(valley_y[pointsMin[1]])
            
            
            ax1.plot(bb[indice1],valley_y[pointsMin[0]],color = 'b', marker='o',label=r'$\min$ $\mu_{\hat{y}(k)_{\textnormal{int}}^{h}}$')
            ax1.plot(bb[indice2],valley_y[pointsMin[1]],color = 'b', marker='o')
            
            
            ax1.axhline(y=np.nanmin(valley_y ), linestyle='--', color = 'b', label=r'$\max$ $C_{\textnormal{int}}^{h}\to T_{\textnormal{int}}^{h}$')
  
            ax1.plot(RW1c+offset, RW2c, color='gray', linewidth=2,alpha=0.35, label=r'C$\to$C')
            valley_indexes = signal.argrelextrema(np.array(RW2c), np.less)
            valley_indexes = valley_indexes[0]
            valley_x = valley_indexes
            valley_y = np.array(RW2c)[valley_indexes]
            
            aa = RW2c.tolist()
            bb = RW1c.tolist()
            
            pointsMin = sorted(range(len(valley_y)), key=lambda sub: valley_y[sub], reverse=False)[:2] 
            indice1 = aa.index(valley_y[pointsMin[0]])
            indice2 = aa.index(valley_y[pointsMin[1]])
            
            
            ax1.plot(bb[indice1],valley_y[pointsMin[0]],color = 'gray', marker='o', label=r'$\min$ $\mu_{\hat{y}(k)}$')
            ax1.plot(bb[indice2],valley_y[pointsMin[1]],color = 'gray', marker='o')
            
            
            ax1.axhline(y=np.nanmin(valley_y ), linestyle='--', color = 'gray', label=r'$\max$ C$\to$C')
            ax1.plot(RW1i+offset, RW2i, color='navy', linewidth=2,  alpha=0.25, label=r'$C_{\textnormal{int}}\to T_{\textnormal{int}}$')
            # Find valleys(min).
            valley_indexes2 = signal.argrelextrema(np.array(RW2i), np.less)
            valley_indexes2 = valley_indexes2[0]
            # Plot valleys.
            valley_x2 = valley_indexes2
            valley_y2 = np.array(RW2i)[valley_indexes2]
            
            aa = RW2i.tolist()
            bb = RW1i.tolist()
            
            pointsMin = sorted(range(len(valley_y2)), key=lambda sub: valley_y2[sub], reverse=False)[:2] 
            indice1 = aa.index(valley_y2[pointsMin[0]])
            indice2 = aa.index(valley_y2[pointsMin[1]])
            
            
            ax1.plot(bb[indice1],valley_y2[pointsMin[0]],color = 'navy', marker='o', label=r'$\min$ $\mu_{\hat{y}(k)_{\textnormal{int}}}$')
            ax1.plot(bb[indice2],valley_y2[pointsMin[1]],color = 'navy', marker='o')
            
            
            
            ax1.axhline(y=np.nanmin(valley_y2 ), linestyle='--', color = 'navy', label=r'$\max$ $C_{\textnormal{int}}\to T_{\textnormal{int}}$')
            
            
            ax1.plot(RW1t+offset, RW2t, color='brown', linewidth=2,  alpha=0.55, label=r'$\textnormal{C}\to \textnormal{T}$')

            valley_indexes2 = signal.argrelextrema(np.array(RW2t), np.less)
            valley_indexes2 = valley_indexes2[0]
            # Plot valleys.
            valley_x2 = valley_indexes2
            valley_y2 = np.array(RW2t)[valley_indexes2]
              
            
            aa = RW2t.tolist()
            bb = RW1t.tolist()
            
            
            pointsMin = sorted(range(len(valley_y2)), key=lambda sub: valley_y2[sub], reverse=False)[:3] 
            
            indice1 = aa.index(valley_y2[pointsMin[0]])
            indice2 = aa.index(valley_y2[pointsMin[1]])
            
            
            ax1.plot(bb[indice1],valley_y2[pointsMin[0]],color = 'brown', marker='o',label=r'$\min$ $\mu_{\hat{y}(k)}$')
            ax1.plot(bb[indice2],valley_y2[pointsMin[1]],color = 'brown', marker='o')
            
            ax1.axhline(y=np.nanmin(valley_y2 ), linestyle='--', color = 'brown', label=r'$\max$ $\textnormal{C}\to \textnormal{T}$')
            ax1.legend(prop={'size': 14})
           
            #ax1.invert_xaxis()
            ax1.set_ylabel(r'$\mu_{\hat{y}^{d}(t-n)[L]}$', fontsize=16)  
            ax1.set_xlabel(r'$k$-lags', fontsize=14)  
            ax1.xaxis.set_label_coords(.5, -.07)
            ax1.yaxis.set_label_coords(-0.075, 0.5)
            ax1.set_xlim([0, 18])

            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax1.legend(loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=True)


    path='results/plots/'
    namefile = 'resRW_%s_N_%d.png'%(test_type,block)

   
    
 
    fig.savefig(path + namefile, dpi=DPI)
    fig.tight_layout()
    
            
          

    plt.show()      

            
        
    
 













    
    
     
    
    
    
        
    

      
        
    
 












