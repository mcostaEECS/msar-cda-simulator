from __future__ import division
import numpy as np
import time
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io
from sklearn.metrics import roc_curve, roc_auc_score

import os
import pyprind
import re
import scipy.io as sio
import datetime
import numpy as np
import time
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io
import pyprind
import math
from ast import literal_eval
import scienceplots


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

    path = 'results/ROC/'
    pathOut = 'results/plots/'
    files_plots = [i for i in os.listdir(path) if i.endswith('.mat')]
    
    PD = []
    PFA = []
    window = []
    Type = []
    pd=[];pfa=[]

    for i in range(len(files_plots)):
        if files_plots[i].find('MSAR_') != -1:
                #print(files_plots[i])
                Test = r'MS-AR($6,1,0$)'
                N = int(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', files_plots[i][:-4])[0])
                K = int(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', files_plots[i][:-4])[1])
                r = scipy.io.loadmat(path+files_plots[i])
       
                pfa_range=r['pfa_range.mat'][0].tolist()
                PD_type=r['PD_type.mat'].tolist()[0]
                FAR_type=r['FAR_type.mat'].tolist()[0]
                
                PD_type= literal_eval(PD_type)
                
                
                FAR_type= literal_eval(FAR_type)  
                
                
        elif files_plots[i].find('MSARkh_') != -1:
                #print(files_plots[i])
                Test = r'MS-AR($18,2,1$)'
                N = int(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', files_plots[i][:-4])[0])
                K = int(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', files_plots[i][:-4])[1])
               
                r = scipy.io.loadmat(path+files_plots[i])
       
                pfa_range=r['pfa_range.mat'][0].tolist()
                PD_type=r['PD_type.mat'].tolist()[0]
                FAR_type=r['FAR_type.mat'].tolist()[0]
                
                PD_type= literal_eval(PD_type)
                
                
                FAR_type= literal_eval(FAR_type)
                
           
        
        elif files_plots[i].find('GPalm2020') != -1:
                   Test = r'AR(n)-GSP\textsuperscript{[15]}'
                   N = int(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', files_plots[i][:-4])[0])
                   K = int(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', files_plots[i][:-4])[1])
                   

                   r = scipy.io.loadmat(path+files_plots[i])
                   pfa_range=r['pfa_range.mat'][0].tolist()
                   PD_type=r['PD_type.mat'].tolist()[0]
                   FAR_type=r['FAR_type.mat'].tolist()[0]

                   PD_type= literal_eval(PD_type)
                   FAR_type= literal_eval(FAR_type)      
                
                
        elif files_plots[i].find('MSARk_') != -1:
                #print(files_plots[i])
                Test = r'MS-AR($18,2,0$)'
                N = int(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', files_plots[i][:-4])[0])
                K = int(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', files_plots[i][:-4])[1])
                
                
                

                r = scipy.io.loadmat(path+files_plots[i])
       
                pfa_range=r['pfa_range.mat'][0].tolist()
                PD_type=r['PD_type.mat'].tolist()[0]
                FAR_type=r['FAR_type.mat'].tolist()[0]
                
                PD_type= literal_eval(PD_type)
                
                
                FAR_type= literal_eval(FAR_type)  
       
 
        elif files_plots[i].find('CNN-GSP') != -1:
                   Test = r'CNN-GSP\textsuperscript{[17]}'
                   N = int(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', files_plots[i][:-4])[0])
                   K = int(re.findall(r'[-+]?(?:\d*\.\d+|\d+)', files_plots[i][:-4])[1])
                   

                   r = scipy.io.loadmat(path+files_plots[i])
                   pfa_range=r['pfa_range.mat'][0].tolist()
                   PD_type=r['PD_type.mat'].tolist()[0]
                   FAR_type=r['FAR_type.mat'].tolist()[0]

                   PD_type= literal_eval(PD_type)
                   FAR_type= literal_eval(FAR_type) 
     
                

                   
                   
        PD.append(PD_type); PFA.append(FAR_type); window.append(N); Type.append(Test) 
        

    plt.rc('text', usetex=True)
    DPI = 600


    SMALL_SIZE = 10
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 12

    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    
 
    pparam2 = dict(xlabel=r'FLOP/sec', ylabel=r'Battery Discharge ($\%$)')

    

    with plt.style.context(['science', 'ieee', 'std-colors']):
            
            #order = [3, 0, 4, 2, 1] 
            fig, ax2 = plt.subplots(figsize=(5,3), dpi=DPI)
            ax2.set_prop_cycle(color=['darkred', 'indianred', 'royalblue', 'cornflowerblue', 'gray'],marker = ['o','*', 'p', 's','<'], linestyle=['dashdot','solid', 'solid', 'solid', 'solid'],alpha=[0.8, 0.4, 0.8, 0.3, 0.3]) #linewidth=2
            ax2.set_prop_cycle(color=['royalblue', 'indianred', 'darkblue','darkred', 'darkviolet',  'indianred'],marker = ['o','*', 'p', 's','<', 'o'])#linewidth=2
            for i in range(len(PD)):
                
                PFA[i] = list(map(lambda i: 0.0001 if i==0 else i, PFA[i]))
                PFA[i] = np.clip(PFA[i],0,1)#[min(x, 0.00001) for x in PFA[i]]
                
                
                far = np.array(PFA[i])
                pd  = np.array(PD[i])
                print(Type[i])
                print(far)
                print(pd)

                # Sort and normalize
                idx = np.argsort(far)
                far_sorted = far[idx]
                pd_sorted = pd[idx]
                far_norm = (far_sorted - far_sorted[0]) / (far_sorted[-1] - far_sorted[0])

                # Compute AUC
                auc_norm = np.trapz(pd_sorted, far_norm)
                print(f"Normalized AUC (MS-AR(18,2,1)) = {auc_norm:.4f}")
        
                PFA[i] = [math.log(y,10) for y in PFA[i]]
                
                
                
                
                [FARx, PDy] = OrderData(PD[i], PFA[i])
                end  = np.argmax(PDy)
                 

                ax2.plot(PFA[i], PD[i], linewidth=2, label=r'%s(AUC=%.4f)'%(Type[i],auc_norm)) # test append(s)
                
                

            ax2.legend(title='Order')
            ax2.set(**pparam2)
            ax2.legend(prop={'size':12})
            ax2.set_ylabel(r'$P_{D}$', fontsize=14)  
            ax2.set_xlabel(r'$\log_{10}$ FAR', fontsize=14)  
            ax2.xaxis.set_label_coords(.5, -.15)
            ax2.set_xlim([-1.8, 0])
            #ax2.set_xlim([0.25, 1.25])
            ax2.set_ylim([0.93, 1.005])
             
            
            ax2.grid(True)
            g1 = ax2.grid(visible=True, which='major', color='k', linestyle='-', alpha=0.45, linewidth=0.5)
            g2 = ax2.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.25, linewidth=0.2)
            ax2.minorticks_on()
            ax2.yaxis.set_tick_params(labelleft=False, labelright=True)
            
            # reordering the labels 
            handles, labels = plt.gca().get_legend_handles_labels() 
            
            # specify order 
            order = [3, 0, 4, 2, 1] 
            order = [0, 1, 2, 3]
            
            
            # pass handle & labels lists along with order as below 
            ax2.legend([handles[i] for i in order], [labels[i] for i in order],facecolor="pink") 


            ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2)
           
            fig.subplots_adjust(bottom=0.2)
            fig.savefig(pathOut + 'Plot_ROC.png', dpi=DPI)
    plt.show()      

        

            
    
    