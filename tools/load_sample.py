import scipy.io as sio
import numpy as np

# --- Plotting --- #
import matplotlib.pyplot as plt


def load_sample(test_type):

    path = 'data/sample/'
    path2 = 'data/sample/targets/'
    
    # images of mission by pair  # Sample_IrefS_AF1.mat
    ImageRead1=sio.loadmat(path+'Sample_IrefS_S1.mat')
    Im1=ImageRead1['ICD']
    ImageRead2=sio.loadmat(path+'Sample_IrefT_S1.mat')
    Im2=ImageRead2['ICD']
    ImageRead3=sio.loadmat(path+'Sample_IrefU_S1.mat')
    Im3=ImageRead3['ICD']
    ImageRead4=sio.loadmat(path+'Sample_IrefV_S1.mat')
    Im4=ImageRead4['ICD']
    ImageRead5=sio.loadmat(path+'Sample_IrefX_S1.mat')
    Im5=ImageRead5['ICD']
    ImageRead6=sio.loadmat(path+'Sample_ItestCT_S1.mat')
    Im6=ImageRead6['ICD']
    ImageRead7=sio.loadmat(path+'Sample_IrefS_K1.mat')
    Im7=ImageRead7['ICD']
    ImageRead8=sio.loadmat(path+'Sample_IrefT_K1.mat')
    Im8=ImageRead8['ICD']
    ImageRead9=sio.loadmat(path+'Sample_IrefU_K1.mat')
    Im9=ImageRead9['ICD']
    ImageRead10=sio.loadmat(path+'Sample_IrefV_K1.mat')
    Im10=ImageRead10['ICD']
    ImageRead11=sio.loadmat(path+'Sample_IrefX_K1.mat')
    Im11=ImageRead11['ICD']
    ImageRead12=sio.loadmat(path+'Sample_ItestCT_K1.mat')
    Im12=ImageRead12['ICD']
    ImageRead13=sio.loadmat(path+'Sample_IrefS_F1.mat')
    Im13=ImageRead13['ICD']
    ImageRead14=sio.loadmat(path+'Sample_IrefT_F1.mat')
    Im14=ImageRead14['ICD']
    ImageRead15=sio.loadmat(path+'Sample_IrefU_F1.mat')
    Im15=ImageRead15['ICD']
    ImageRead16=sio.loadmat(path+'Sample_IrefV_F1.mat')
    Im16=ImageRead16['ICD']
    ImageRead17=sio.loadmat(path+'Sample_IrefX_F1.mat')
    Im17=ImageRead17['ICD']
    ImageRead18=sio.loadmat(path+'Sample_ItestCT_F1.mat')
    Im18=ImageRead18['ICD']
    ImageRead19=sio.loadmat(path+'Sample_IrefS_AF1.mat')
    Im19=ImageRead19['ICD']
    ImageRead20=sio.loadmat(path+'Sample_IrefT_AF1.mat')
    Im20=ImageRead20['ICD']
    ImageRead21=sio.loadmat(path+'Sample_IrefU_AF1.mat')
    Im21=ImageRead21['ICD']
    ImageRead22=sio.loadmat(path+'Sample_IrefV_AF1.mat')
    Im22=ImageRead22['ICD']
    ImageRead23=sio.loadmat(path+'Sample_IrefX_AF1.mat')
    Im23=ImageRead23['ICD']
    ImageRead24=sio.loadmat(path+'Sample_ItestCT_AF1.mat')
    Im24=ImageRead24['ICD']
    
    
    # tragets position by mission
    tp3=sio.loadmat(path2+'F1.mat')
    tp3=tp3['F1']
    

    
    if test_type == 'MSAR' or test_type == 'MSARk' or  test_type == 'MSARkh' or  test_type == 'MSARh':
        
          par=[[Im19, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im20, Im21, Im22, Im23, Im24, Im14, Im15, Im16, Im17, Im18, tp3,'F1', Im13, 13],
               [Im2,Im1, Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im19, Im20, Im21, Im22, Im23, Im24, Im13, Im15, Im16, Im17, Im18,tp3,'F1', Im14, 14],
               [Im9, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8,  Im10, Im11, Im12, Im19, Im20, Im21, Im22, Im23, Im24, Im14, Im13, Im16, Im17, Im18,tp3,'F1', Im15,15],
               [Im22, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im19, Im20, Im21,  Im23, Im24,Im14, Im15, Im13, Im17, Im18, tp3,'F1', Im16, 16],
               [Im5, Im1,Im2,Im3,Im4,  Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im19, Im20, Im21, Im22, Im23, Im24, Im14, Im15, Im16, Im13, Im18,tp3,'F1', Im17, 17],
               [Im12, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11,  Im19, Im20, Im21, Im22, Im23, Im24, Im14, Im15, Im16, Im17, Im13,tp3,'F1', Im18, 28]]
    

    else:
        
        print('Invalid test')
    
        
        
            
    
        
    return par

# res =load_data('MSAR')[0]
# tp=res[23]; tp =  np.fliplr(tp); TP =res[24]

# Itest = res[21]

# print(Itest.shape)


# fig = plt.figure('test')
# plt.suptitle('REF / TEST / Change')

# ax = fig.add_subplot(1, 1, 1)
# plt.imshow(Itest, cmap = plt.cm.gray)
# plt.axis("off")



# plt.show() 
            
    

    
    
