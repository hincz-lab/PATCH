import numpy as np
import scipy
from scipy import stats


inny=open('my_out_file.txt','r')
fff=inny.readlines()



##############FIND MEAN AND MAX######################
vsdict={}
patchnum=[] ##this has nothing to do with patches 
for ii in range(len(fff)):
    if fff[ii].startswith("fgp"): ##change this to identify individual folds
        vsdict[fff[ii].strip()]=[]
        vs=fff[ii].strip()
        partches=fff[ii+7]
        patchnum.append(partches)
        
    elif fff[ii].startswith('Epoch 25: val_accuracy improved'):
        onestep=fff[ii].strip().split(',')
        vsdict[vs].append(float(onestep[0].strip().split(' ')[7])) 

    elif fff[ii].startswith('Epoch 25:'):
        
        vsdict[vs].append(float(fff[ii].strip().split(' ')[7]))
        
    
for eyedem in vsdict.keys():
    m=np.mean(vsdict[eyedem])
    mx=max(vsdict[eyedem])
    medd=np.median(vsdict[eyedem])
    print(eyedem,m,mx)#,medd)

for item in patchnum:
    print(item.strip())
    
  
###############################--END--######################
