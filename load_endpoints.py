# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:42:20 2019

@author: gham
"""


import os
import numpy as np
import nibabel as nib
import bisect
import time
import pickle
import pandas as pd
import copy

import matplotlib.pyplot as plt



# Loading endpoints and label
with open("endpoints-shorder=6-maxangle=30-gfa=0.20-BSdiv.pkl",'rb') as f:
    endpoints=pickle.load(f)
with open("reduced_label-shorder=6-maxangle=30-gfa=0.20-BSdiv.pkl",'rb') as g:
    label=pickle.load(g)


# Maps the endpoint coordinates into the labels
endpoints_label_map=np.zeros([endpoints.shape[0],endpoints.shape[1]])
print("Creating endpoints_label_map")
for i in range(endpoints_label_map.shape[0]):
    if i==endpoints_label_map.shape[0]//2:
        print("Creating endoints_label_map Halfway done")
    a=np.array(np.round(((endpoints[i][0])+np.array([-90,126,72]))*np.array([-1/1.25,1/1.25,1/1.25])),dtype='int32')
    b=np.array(np.round(((endpoints[i][1])+np.array([-90,126,72]))*np.array([-1/1.25,1/1.25,1/1.25])),dtype='int32')
    
    endpoints_label_map[i][0]=label[a[0],a[1],a[2]]
    endpoints_label_map[i][1]=label[b[0],b[1],b[2]]
print("Done creaing endpoints_label_map")


# load label informations
aseg_dir="C:\\Users\\gham\\Desktop\\Human Brain\\Data\\102109\\102109_3T_Structural_preproc\\102109\\T1w\\102109\\stats\\"
aseg_stat_file="aseg.stats"
headers="Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange".split(' ') # for aseg
aseg_stat=pd.read_csv(aseg_dir+aseg_stat_file,sep='\s+',comment='#', names=headers)
                      
cortex_stats='cortex.stats'
headers="No. Labelname R G B A".split(" ")
cortex_stat=pd.read_csv(aseg_dir+cortex_stats,sep='\s+',comment='#', names=headers)


# add aseg stat                      
id_to_name={i:j for i,j in zip(aseg_stat['SegId'],aseg_stat['StructName']) if i!=j}
id_to_name[16]='Medulla'
id_to_name[90]='Pons'
id_to_name[120]='Midbrain'
# add  cortex stats
for i,j in zip(cortex_stat['No.'], cortex_stat['Labelname']):
    id_to_name[i]=j


label_map={}
for i in id_to_name.keys():
    label_map[i]={}
    for j in id_to_name.keys():
        label_map[i][j]=0
for e in endpoints_label_map:
    if e[0] not in [2,41,7,46,0]:
        if e[1] not in [2,41,7,46,0]:
            label_map[e[0]][e[1]]+=1
            
df=pd.DataFrame(label_map)

# 16 to all
for i,j in zip(df[16].index,df[16]):
    if i not in [2,41,7,46,0] and j>10:
        print(id_to_name[16], " to ", id_to_name[i], " : ",j)

for i,j in zip(df[90].index,df[90]):
    if i not in [2,41,7,46,0] and j>10:
        print(id_to_name[90], " to ", id_to_name[i], " : ",j)
        
for i,j in zip(df[120].index,df[120]):
    if i not in [2,41,7,46,0] and j>10:
        print(id_to_name[120], " to ", id_to_name[i], " : ",j)
        
for i in id_to_name.keys():
    if i not in [2,41,7,46,0,72]:
        for j in id_to_name.keys():
            if j not in [2,41,7,46,0,72]:
                amount=label_map[i][j]
                if amount>5 and j==120:
                    print("from ",id_to_name[i]," to ",id_to_name[j]," : ",amount)

#combine both directions
new_label_map=copy.deepcopy(label_map)
for i in id_to_name.keys():
    for j in id_to_name.keys():
        new_label_map[i][j]+=label_map[j][i]

with open ("bi-directional_label_map-shorder=6-maxangle=30-gfa=0.20-BSdiv.pkl",'wb') as l:
    pickle.dump(new_label_map,l)



#label[:,0] = reduced_size_label[]
#label[:,1] = label_file[:,1,:]

#print(label.shape)








