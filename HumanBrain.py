# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:34:58 2019

@author: gham
"""

import os
import numpy as np
import nibabel as nib
import bisect
import time
import pickle

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.tracking.streamline import Streamlines

import matplotlib.pyplot as plt

# Neuroscience
import dipy
from dipy.core.gradients import gradient_table


# Diffusion
path_d='C:\\Users\\gham\\Desktop\\Human Brain\\Data\\102109\\102109_3T_Diffusion_preproc\\'
datafile_d='102109\\T1w\\Diffusion\\data.nii.gz'
#T1w_acpc_dc_restore='102109\\T1w\\T1w_acpc_dc_restore_1.25.nii.gz'

bvals=np.genfromtxt(path_d+"102109\\T1w\\Diffusion\\bvals",dtype='int')
bvecs=np.genfromtxt(path_d+"102109\\T1w\\Diffusion\\bvecs")
bvecs=np.array([[i,j,k] for i,j,k in zip(*bvecs)],dtype='float64')
gtab=gradient_table(bvals,bvecs=bvecs)
assert gtab.bvecs.shape==bvecs.shape
assert gtab.bvals.shape==bvals.shape

# Structural
path_s="C:\\Users\\gham\\Desktop\\Human Brain\\Data\\102109\\102109_3T_Structural_preproc\\102109\\T1w\\"
datafile_aparc_aseg="aparc+aseg.nii.gz"



def Analyze(img_d_path,img_s_path,gtab):
    
    # For fiber tracking, 3 things are needed
    # 1. Method for getting directions
    # 2. Method for identifying different tissue types
    # 3. seeds to begin tracking from
        
    print_info=False
    if print_info:
        print("============================  Diffusion  ============================")
        print(img_d)
        print("============================ Structural =============================")
        print(img_s)
        print("Labels:", np.unique(img_s.get_data()).astype('int'))
    
    # Load images
    img_d=nib.load(img_d_path)
    img_s=nib.load(img_s_path)
        

    # Resize the label (img_s)
    # 0. create an empty array the shape of diffusion image, without
    #    the 4th dimension 
    # 1. Convert structural voxel coordinates into ref space (affine)
    # 2. Convert diffusion voxel coordinates into ref space (affine)
    # 3 For each diffusion ref coordinate,
    #   find the closest structural ref coordinate
    # 4. find its corresponding label, then input it to the empty array
    
    print_info_2=True
    if print_info_2:
        #print(img_d.get_data().shape[])
        print(img_d.affine)
        print(img_s.affine)
        print(img_s._dataobj._shape)
        print(img_d._dataobj._shape)
        
    img_d_shape_3D=[j for i,j in enumerate(img_d.dataobj._shape) if i<3]
    img_s_shape_3D=img_s.dataobj._shape

    #raise ValueError(" ")
    
    
    img_d_affine=img_d.affine
    img_s_affine=img_s.affine
    
    img_s_data=img_s.get_data()
    img_d_data=img_d.get_data()
    
    Vox_coord_s_i = np.arange(img_s_shape_3D[0])
    Vox_coord_s_j = np.arange(img_s_shape_3D[1])
    Vox_coord_s_k = np.arange(img_s_shape_3D[2])
    Ref_coord_s_i = Vox_coord_s_i*img_s_affine[0,0]+img_s_affine[0,3]
    Ref_coord_s_j = Vox_coord_s_j*img_s_affine[1,1]+img_s_affine[1,3]
    Ref_coord_s_k = Vox_coord_s_k*img_s_affine[2,2]+img_s_affine[2,3]
    #print(Ref_coord_s_j)

    reduced_size_label=np.zeros(img_d_shape_3D)

    for i in range(img_d_shape_3D[0]):
        for j in range(img_d_shape_3D[1]):
            for k in range(img_d_shape_3D[2]):
                # convert to reference coordinate
                ref_coord_i=i*img_d_affine[0,0]+img_d_affine[0,3]
                ref_coord_j=j*img_d_affine[1,1]+img_d_affine[1,3]
                ref_coord_k=k*img_d_affine[2,2]+img_d_affine[2,3]
                
                min_i_ind=bisect.bisect_left(np.sort(Ref_coord_s_i),ref_coord_i)
                min_j_ind=bisect.bisect_left(Ref_coord_s_j,ref_coord_j)
                min_k_ind=bisect.bisect_left(Ref_coord_s_k,ref_coord_k)
                #print(min_i_ind,min_j_ind,min_k_ind)
                #print(img_s_data[260-1-min_i_ind][311-1-min_j_ind][260-1-min_k_ind])
                #reduced_size_label[i][j][k]=img_s_data[260-1-min_i_ind][311-1-min_j_ind][260-1-min_k_ind]
                reduced_size_label[i][j][k]=img_s_data[260-1-min_i_ind,min_j_ind,min_k_ind]
    print("Label image reduction successful")
    
    # Divide Brainstem
    #msk_Midbrain
    yy,xx,zz=np.meshgrid(np.arange(174),np.arange(145),np.arange(145))
    
    pon_midbrain_msk=yy>(-115/78)*zz+115
    midbrain_msk=zz>48

    BS_msk=reduced_size_label==16
    reduced_size_label_BS_seg=np.copy(reduced_size_label)
    reduced_size_label_BS_seg[BS_msk*pon_midbrain_msk]=90
    reduced_size_label_BS_seg[BS_msk*midbrain_msk]=120
    
    
    
    plt.figure(figsize=[11,8.5])
    msk=reduced_size_label>200
    temp_reduced_size_label=np.copy(reduced_size_label_BS_seg)
    temp_reduced_size_label[msk]=0
    plt.imshow(temp_reduced_size_label[72,:,:],origin='lower')
    
    msk=reduced_size_label==16
    temp_reduced_size_label=np.copy(reduced_size_label)
    temp_reduced_size_label[~msk]=0
    plt.figure(figsize=[11,8.5])
    plt.imshow(temp_reduced_size_label[72,:,:],origin='lower')
    
    
    #print("image display complete")
    #input1=raw_input("stopping")
    T1_path="C:\\Users\\gham\\Desktop\\Human Brain\\Data\\102109\\102109_3T_Diffusion_preproc\\102109\\T1w\\"
    T1_file="T1w_acpc_dc_restore_1.25.nii.gz"
    T1=nib.load(T1_path+T1_file)
    T1_data=T1.get_data()
    plt.figure(figsize=[11,8.5])
    plt.imshow(T1_data[72,:,:],origin='lower')
    plt.show()
    
    # implement the modified label
    reduced_size_label=reduced_size_label_BS_seg
    #raise ValueError("========== Stop =============")
    
    
    
    #White matter mask
    left_cerebral_wm=reduced_size_label==2
    right_cerebral_wm=reduced_size_label==41
    cerebral_wm=left_cerebral_wm+right_cerebral_wm
    left_cerebellum_wm=reduced_size_label==7
    right_cerebellum_wm=reduced_size_label==46
    cerebellum_wm=left_cerebellum_wm+right_cerebellum_wm
    
    CC=np.zeros(reduced_size_label.shape)
    for i in [251,252,253,254,255]:
        CC+=reduced_size_label==i
    
    left_cortex=np.zeros(reduced_size_label.shape)
    for i in np.arange(1000,1036):
        left_cortex+=reduced_size_label==i
    right_cortex=np.zeros(reduced_size_label.shape)
    for i in np.arange(2000,2036):
        right_cortex+=reduced_size_label==i
        
    extra=np.zeros(reduced_size_label.shape)
    for i in [4,5,8,10,11,12,13,14,15,16,90,120,17,18,24,26,28,30,31,43,44,46,47,49,50,51,52,53,54,58,60,62,63,77,80,85]:
        extra+=reduced_size_label==i
    #for i in np.arange(1001,1035):
    #    extra+=reduced_size_label==i
    
    
    wm=cerebral_wm+cerebellum_wm+CC+extra+left_cortex+right_cortex
    
    
    
    
    #seed_mask1=np.zeros(reduced_size_label.shape)
    #for i in [16]:
    #    seed_mask1+=reduced_size_label==i
    #seed_mask2=np.zeros(reduced_size_label.shape)
    
    #seed_mask=seed_mask1+seed_mask2
    
    #seed_mask=(reduced_size_label==16)+(reduced_size_label==2)+(reduced_size_label==41)
    #seeds = utils.seeds_from_mask(seed_mask, density=1, affine=img_d_affine)
    seeds = utils.seeds_from_mask(wm, density=1, affine=img_d_affine)
    
    # Constrained Spherical Deconvolution
    #reference: https://www.imagilys.com/constrained-spherical-deconvolution-CSD-tractography/
    csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
    csd_fit = csd_model.fit(img_d_data, mask=wm)

    print("CSD model complete")
    
    
    # reconstruction
    from dipy.reconst.shm import CsaOdfModel

    csa_model = CsaOdfModel(gtab, sh_order=6)
    gfa = csa_model.fit(img_d_data, mask=wm).gfa
    classifier = ThresholdTissueClassifier(gfa, .25)    
# =============================================================================
#     import dipy.reconst.dti as dti
#     from dipy.reconst.dti import fractional_anisotropy
#     tensor_model = dti.TensorModel(gtab)
#     tenfit=tensor_model.fit(img_d_data,mask=wm) #COMPUTATIONALL INTENSE
#     FA=fractional_anisotropy(tenfit.evals)
#     classifier=ThresholdTissueClassifier(FA,.1) # 0.2 enough?
# =============================================================================

    print("Classifier complete")
    
    # Probabilistic direction getter
    from dipy.direction import ProbabilisticDirectionGetter
    from dipy.data import small_sphere
    from dipy.io.streamline import save_trk
    
    fod = csd_fit.odf(small_sphere)
    pmf = fod.clip(min=0)
    prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=75.,
                                                    sphere=small_sphere)
    streamlines_generator = LocalTracking(prob_dg, classifier, seeds, img_d_affine, step_size=.5)
    save_trk("probabilistic_small_sphere.trk", streamlines_generator, img_d_affine, reduced_size_label.shape)
    
    astreamlines=np.array(list(streamlines_generator))
    endpoints=np.array([st[0::len(st)-1] for st in astreamlines if len(st)>1] )
    
    print(endpoints)
    with open('endpoints-shorder=6-maxangle=75-gfa=0.25-BSdiv-v3.pkl','wb') as f:
        pickle.dump(endpoints,f)
    with open("reduced_label-shorder=6-maxangle=75-gfa=0.25-BSdiv-v3.pkl","wb") as g:
        pickle.dump(reduced_size_label,g)
    #with open('endpoints.txt','w') as f:
    #    f.write(endpoints)
# =============================================================================
#     M, grouping = utils.connectivity_matrix(streamlines_generator, np.array(reduced_size_label,dtype='int32'), affine=img_d_affine,
#                                         return_mapping=True,
#                                         mapping_as_streamlines=True)
#     print(M)
#     print(grouping)
# =============================================================================
    #M, grouping = utils.connectivity_matrix(streamlines, intlabels,affine=img_d_data.affine,
    #                                    return_mapping=True,
    #                                    mapping_as_streamlines=True)
    #print(M)
    
    
begin=time.time()
Analyze(path_d+datafile_d,path_s+datafile_aparc_aseg,gtab)
end=time.time()

print("time elapsed: ",end-begin,' s')

















