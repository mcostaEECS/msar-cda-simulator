from __future__ import division
import numpy as np
import time
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io
import os
from load_data import load_data
import pyprind
import re
import scipy.io as sio
from Classifier import Classifier
import datetime
import pyprind
import time
import psutil
import datetime


def ROC(th, window, test_type):
    
    detected_targets=[]
    falseAlarm = []
    mission_targets = []
    RES = []


    if test_type == 'MSAR':
        path = 'results/MSAR/'
    elif test_type == 'MSARk':
        path = 'results/dataMSARk/'
    elif test_type == 'MSARkh':
        path = 'results/dataMSARkh/'
   

    files_icd = [i for i in os.listdir(path) if i.endswith('.mat')]
    nroPairs = len(files_icd)

    
    #bar = pyprind.ProgBar(th, monitor=True, title=test_type)
    campaign = 'Test_%s_th_%.2f_window_%d'%(test_type, th, window)
    
    path2 = 'results/ROC/'
    nameFile = 'Table_ROC_%s_window_%d'%(test_type, window) 
    id = path2+nameFile+'.txt'

    START = time.time()

    bar = pyprind.ProgBar(nroPairs, monitor=True, title=campaign)
    with open(id, 'a') as f:

        for i in range(len(files_icd)):
                if files_icd[i].find(test_type) != -1:
                    
                    pair = int(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", files_icd[i])[-1])
                    
                    
          
                    par=load_data(test_type)[pair]
                    
                    #tp=par[18]; tp =  np.fliplr(tp); TP =par[19]; #pair = par[-1]
                    tp=par[23]; tp =  np.fliplr(tp); TP =par[24]
                    
                    
                    ImageRead1=sio.loadmat(path+files_icd[i])
                    ICD=ImageRead1['ICD']
                    
                    t = ICD.mean(axis=0).mean(axis=0)
  
    
                    [dt, fa, mt]=Classifier(ICD,tp,th, test_type)
                    pd_partial = dt/25; far_partial = fa/6
                    res = 'Pair_'+str(pair)+';'+'detected_targets_'+str(dt)+';'+'pd_'+str(pd_partial)+';'+'false_target_'+str(fa)+';'+'far_'+str(far_partial)+';'+'missed_target_'+str(25-dt)
                    f.write(res+'\n')
                    detected_targets.append(dt)
                    falseAlarm.append(fa)
                    mission_targets.append(mt)
                    RES.append(res)
                    END = time.time()
                    time_ROC = END - START
                bar.update()
    

        PD= np.sum(detected_targets)/600
        FAR = np.sum(falseAlarm)/(144)
        f.write('PD_'+str(PD)+';'+'FAR_'+str(FAR)+';'+'threshold_'+str(th)+'\n')
        f.write('Finish threshold\n')


    return PD, FAR, RES
    


if __name__ == "__main__":

       
    
    
    test_type = ['MSAR',  'MSARk', 'MSARkh']
    test_type  = test_type[0]
    

    

    N = 250
    K = 9
    
    campaign = 'Campaign_%s_N_%d_K_%d'%(test_type, N, K)
    pathOut = 'results/ROC/'
    id = pathOut+campaign+'.mat'

    pfa_min = 0.5; pfa_max = 2.6
    
    #pfa_min = 0.5; pfa_max = 3.6
    
    
    
    
    
    
    pfa_range = np.arange(pfa_min, pfa_max,0.25)#[::-1]
    

 
    
    
   
   
    start_time = time.perf_counter()

    PD=[]
    FAR=[]
    RES =[]

    nroTh = len(pfa_range)

    # bar = pyprind.ProgBar(nroTh, monitor=True, title=campaign)

    for i in range(len(pfa_range)):
            [pd, far, Res]= ROC(pfa_range[i], N, test_type)


            
            PD.append(pd)
            FAR.append(far)
            RES.append(Res)  # .mat

            # bar.update()

 
    finish_time = time.perf_counter()
    print(f"Program finished in {(finish_time - start_time):.3f} seconds")

    results={}
    results['test_type.mat']=test_type
    results['pfa_range.mat']=pfa_range
    results['k_range.mat']=K
    results['N_range.mat']=N
    results['PD_type.mat']=" ".join(str(x) for x in [PD])
    results['FAR_type.mat']=" ".join(str(x) for x in [FAR])
    results['Detection_pair.mat']=" ".join(str(x) for x in [RES])
        
    scipy.io.savemat(id,results) 





   


            
    
    