# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:34:07 2019

@author: gham
"""

import os
import numpy as np
import nibabel as nib
import bisect
import time
import pickle
import pandas as pd

import matplotlib.pyplot as plt





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
                        
    
# shorder, maxangle, gfa
p1=["6" , "45"  , "0.25", ""]
p2=["6" , "45"  , "0.25", "-v2"]
p3=["6" , "45"  , "0.25", "-v3"]
#p3=["6" , "60"  , "0.25"]
#p4=["6" , "75"  , "0.25"]
#p5=["6" , "30"  , "0.20"]  
#p6=["6" , "30"  , "0.30"]  


def analyze(p):
    path='C:\\Users\\gham\\Desktop\\Human Brain\\'
    data_file="bi-directional_label_map-shorder="+p[0]+"-maxangle="+p[1]+"-gfa="+p[2]+"-BSdiv"+ p[3] + ".pkl"

    with open (path+data_file,'rb') as f:
        data=pickle.load(f)
    
    Master=pd.DataFrame(    data[16].values()  ,    columns=['Medulla'])
    Pons=pd.DataFrame(      data[90].values()  ,    columns=['Pons'])
    Midbrain=pd.DataFrame(  data[120].values() ,    columns=['Midbrain'])
    
    Master=pd.merge(Master, Pons, how='outer',left_on='Medulla',right_on='Pons',left_index=True, right_index=True)
    Master=pd.merge(Master, Midbrain, how='outer',left_on='Medulla',right_on='Midbrain',left_index=True, right_index=True )
    
    labels_id=pd.DataFrame([i for i in data[16].keys()],  columns=['id'])
    labels_name=pd.DataFrame([id_to_name[i] for i in data[16].keys()],  columns=['name'])  
    Master=labels_name.join(Master)
    Master=labels_id.join(Master)
    
    remove=[2,7,41,46,8,47]
    msk=Master['id']==0
    for i in remove:
        msk+=Master['id']==i
    
    return Master[~msk]

for p in [p1,p2, p3]:
    MPM = analyze(p)
    MPM['Medulla'][MPM['name']=='Medulla']=0
    MPM['Pons'][MPM['name']=='Pons']=0
    MPM['Midbrain'][MPM['name']=='Midbrain']=0
    print("sh_order= ",p[0], " max_angle= ",p[1], " gfa= ",p[2])
    print(MPM)
    MPM.to_csv("Connectivity_sh_order= "+p[0]+ " max_angle= "+p[1]+" gfa= "+p[2] + p[3] +  ".csv")








# =============================================================================
# 
# 
# Master=pd.DataFrame()                        
# def write_to_column(filepath):
#     
#     with open (filepath,'rb') as f:
#         data=pickle.load(f)
# 
#     Series1={}
#     for i in id_to_name.keys():
#         if i not in [2,7,41,46,0,8,47]:
#             for j in id_to_name.keys():
#                 if j not in [2,7,41,46,0,8,47]:
#                     amount=data[i][j]
#                     if i==16 :
#                         Series1[id_to_name[i]+" and "+id_to_name[j]]=amount
#                         #print("between ",id_to_name[i]," and ",id_to_name[j]," :",amount)
#     print(Series1)
#     Master["Medulla"]=Series1
# 
# write_to_column(path+data_file)
# =============================================================================





