from __future__ import division
import numpy as np
import time
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io
#from GLRT import GLRT
import os
#from LMP import LMP
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



def ROC(th, window, test_type):
    
    detected_targets=[]
    falseAlarm = []
    mission_targets = []
    RES = []


    if test_type == 'ARn3':
        path = '/home/marcello-costa/workspace/kthSim/Output/Data3MCnAR1/'
    elif test_type == 'ARn3opt':
        path = '/home/marcello-costa/workspace/kthSim/Output/Data3MCnAR1opt/'
    elif test_type == 'ARn1':
        path = '/home/marcello-costa/workspace/kthSim/Output/Data1MCnAR1/'
    elif test_type == 'ARn2':
        path = '/home/marcello-costa/workspace/kthSim/Output/Data2MCnAR1/'
    elif test_type == 'ARn3opt2':
        path = '/home/marcello-costa/workspace/kthSim/Output/Data3MCnAR1opt2/'
         

    files_icd = [i for i in os.listdir(path) if i.endswith('.mat')]
    nroPairs = len(files_icd)

    
    #bar = pyprind.ProgBar(th, monitor=True, title=test_type)
    campaign = 'Test_%s_th_%.2f_window_%d'%(test_type, th, window)
    
    path2 = '/home/marcello-costa/workspace/kthSim/Output/ROC/'
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
                 
                    
                    # if t > -0.0002:
                    #     th = th
                        
                    # else:
                    #     th = th/5
                    
                   
                    # if pair==0 or pair==4 or pair==8 or pair==12 or pair==16 or pair==20:
                    #     OUT=np.pad(ICD, ((500,0),(0,0)), mode='constant')
                    #     OUT2=np.pad(OUT, ((0,2000),(0,0)), mode='constant') 
                    # elif pair==1 or pair==5 or pair==9 or pair==13 or pair==17 or pair==21:
                    #     OUT=np.pad(ICD, ((250,0),(0,0)), mode='constant') 
                    #     OUT2=np.pad(OUT, ((0,2250),(0,0)), mode='constant') 
                    # elif pair==2 or pair==6 or pair==10 or pair==14 or pair==18 or pair==22:
                    #     OUT=np.pad(ICD, ((2000,0),(0,0)), mode='constant')
                    #     OUT2=np.pad(OUT, ((0,500),(0,0)), mode='constant') 
                    # elif pair==3 or pair==7 or pair==11 or pair==15 or pair==19 or pair==23:
                    #     OUT=np.pad(ICD, ((2250,0),(0,0)), mode='constant')
                    #     OUT2=np.pad(OUT, ((0,250),(0,0)), mode='constant')  
                        
                    
                    
                
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

       
    
    
    test_type = ['ARn1', 'ARn2', 'ARn3',  'ARn3opt', 'ARn3opt2']
    test_type  = test_type[4]
    

    

    N = 250
    K = 9
    
    rep=10
    
    # REP = []
    # Hatt = []; Rwkt=[]; Rwmt=[]; Itestt=[]; Icdt=[]; Tht=[]
    # Hatc = []; Rwkc=[]; Rwmc=[]; Itestc=[]; Icdc=[]; Thc=[]
    # Hati = []; Rwki=[]; Rwmi=[]; Itesti=[]; Icdi=[]; Thi=[]
    # Hath = []; Rwkh=[]; Rwmh=[]; Itesth=[]; Icdh=[]; Thh=[]
    
    for r in range(rep):
    
        campaign = 'Campaign_%s_N_%d_K_%d_rep_%d'%(test_type, N, K,r)
        pathOut = '/home/marcello-costa/workspace/kthSim/Output/ROC/'
        id = pathOut+campaign+'.mat'

        pfa_min = 0.5; pfa_max = 3.6
        
    
        
        
        
        
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



       
    # colormap = np.array(['b', 'c', 'g', 'k', 'm', 'darkorange', 'blueviolet', 'lime','darkorange','blueviolet','orchid','cyan'])
    # markermap = np.array(['X', 's', 'd', 'X', '*', 'p', 'P', 'h', 'H', 'x', '8'])
    # alphamap = np.array([0.6, 0.2, 0.3, 0.5, 0.6])
    # linestyles = ['solid', 'dashed']

    # f = plt.figure(1)
    # s = f.add_subplot(1,1,1)
    # [FARx, PDy] = OrderData(FAR, PD)
    # end  = np.argmax(PDy)
    # s.plot(FARx[:end+1], PDy[:end+1],colormap[0], marker=markermap[0],  linestyle=linestyles[0],label=r'Window=%d'%N) # test append(s)
    # s.set_title('PD Vs FAR')
    # s.set_xlabel('FAR')
    # s.set_ylabel('PD')
    # s.legend(loc=2,prop={'size':8})
    # s.grid()
    # plt.show()
            

   


            
    
    