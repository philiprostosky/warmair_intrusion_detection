#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import netCDF4 as nc
from pyhdf.SD import SD, SDC
from pyresample import image,geometry,utils,kd_tree
from pyresample import create_area_def
import copy
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import glob
import os.path
import os


#Main functions

def dfs_area(A, start):
    '''
    function to detected continuous shapes within a masked array. Core function for 
    warming wave detection algorithm. A = 2d array containing 1 or 0. start = starting entry for function
    '''
    graph = {}
    res = np.zeros_like(A)
    for index, x in np.ndenumerate(A):
        graph[index] = x
    visited, stack = set(), [start]

    while stack:
        vertex = stack.pop()
        x, y = vertex[0], vertex[1]
        if vertex not in visited and graph[vertex] == True:
            visited.add(vertex)
            if x < A.shape[0]-1:
                stack.append((x+1, y))
            if x > 1:
                stack.append((x-1, y))
            if y < A.shape[1]-1:
                stack.append((x, y+1))
            if y > 1:
                stack.append((x, y-1))
    res[tuple(np.array(list(visited)).T)] = 1

    return res

def iceedge_mask(sic,limit=65):
    '''
    checks if ice egde (sic<limit) has moved during warming wave and masks it out. 
    '''
    sic0=np.zeros((sic.shape[1],sic.shape[2]))
    for ii in np.arange(sic.shape[0]):
        flag=sic[ii,:,:]>limit
        sic0[flag]=sic0[flag]+1
    flag=np.broadcast_to(np.logical_and(sic0>0,sic0<np.nanmax(sic0)),sic.shape)
    sic[flag]=np.nan
    return sic


def detect_warming_wave(t2m,sic,th,landmask=np.array([]),ende=-10,vlow=4,delta=4,smin=3,sic_min=65,area_old=np.array([])):
    '''
    function to detect and process shapes of individual warming waves.
    t2m = timeseries of 2m temperature (30 to 40 days recommended)
    sic = timeseries of sea ice concentration from various products
    th = temperature threshold determining a warming event (e.g., 265K or 270K,...)
    landmask = area to be masked out (1==area to be excluded). It is recomended to mask out areas with frequent polynia openings
    ende= last day of the period until which a warming wave is detected. recommended:10 to 15, otherwise algorithm might not work 
    accurately
    vlow= minimum difference between peak of warming wave and the follwing 10 days. (if vlow too low = no wave)
    delta= minimum difference between peak and mean of t2m 
    smin=minimum of days required prior to warming wave. Should be at least 3 to estimate impact of warming 
    on sic
    sic_min =minimum sea ice concentration to be considered. From testing, "wrong" sic rarely drops below 65% 
    during  a warming event, lower walues are rather lead/polynia openings due to dynamics
    area_old=area detected previously (that means affected by previous warming waves) --> if the algorithms is used to process a season or more, use 
    the function: optimize_waves to find the optimal time window for the warming wave
    
    returns areas of detected waves within the timeperiod (+area of significant SIC reduction within the warming 
    waves)
    
    General notes: To obtaion best results the detection algorithm should be applied to a chunk of 30-40 days 
    of data. The algorithm should be applied several times using a sliding window of ~5 days
    (not much larger than "smin") to capture all warming waves within a time period. To not douple detect warming 
    waves previous detected waves should be stored in "area_old" for 2 cycles (10 days)
    '''
    if sic.ndim <4:
        sic=np.reshape(1,sic.shape[0],sic.shape[1],sic.shape[2])
    if landmask.shape[0]>0:
        flag=np.broadcast_to(landmask==1,t2m.shape)
        t2m[flag]=np.nan
    if area_old.shape[0]>0:
        for ii in np.arange(area_old.shape[0]):
            flag=np.broadcast_to(area_old[ii,:,:]==1,t2m.shape)
            t2m[flag]=np.nan
    sicmin=np.nanmin(sic,axis=(0,1))
    flag=np.broadcast_to(sicmin<sic_min,t2m.shape)
    t2m[flag]=np.nan  
    diff=np.nanmax(t2m[:ende,:,:],axis=(0))-np.nanmean(t2m[:ende,:,:],axis=(0))
    flag=np.broadcast_to(diff<delta,t2m.shape)
    t2m[flag]=np.nan
    flag=np.logical_and(np.nanmax(t2m[:ende,:,:],axis=(0))>th[0], # within warming limits
                        np.nanmax(t2m[:ende,:,:],axis=(0))<=275) 
    t2m_zw=np.where(flag, t2m,np.nan) 
    flag=np.isnan(t2m_zw)
    t2m_zw[flag]=-1000
    test=np.nanargmax(t2m_zw[:ende,:,:],axis=0)
    a1,a2=np.indices(test.shape)
    test2=np.array([]).reshape(0,test.shape[0],test.shape[1])
    for ii in np.arange(10):
        test2=np.concatenate((test2,t2m_zw[test+ii,a1,a2].reshape(1,test.shape[0],test.shape[1])),axis=0)
    test3=np.nanmin(test2,axis=0)    
    test3[test3<-999]=np.nan
    flag=t2m_zw[test,a1,a2]-test3<vlow
    zw=np.broadcast_to(flag, t2m_zw.shape)
    t2m_zw2=copy.copy(t2m_zw)
    t2m_zw2[zw]=np.nan
    t2m_zw2[t2m_zw2<-999]=np.nan
    flag=test<=smin
    test[flag]=0
    flag=np.broadcast_to(test==0,t2m_zw2.shape)
    t2m_zw2[flag]=np.nan
    t2m_zw2[t2m_zw2<-999]=np.nan
    t2m_zw[t2m_zw<-999]=np.nan
    t2m_zw3=copy.copy(t2m_zw2)
    diff=np.nanmax(t2m_zw2[:-15,:,:],axis=(0))-np.nanmean(t2m_zw2[:-15,:,:],axis=(0))
    flag=np.broadcast_to(diff<4,t2m_zw2.shape)
    t2m_zw3[flag]=np.nan
    area_old=t2m_zw3[0,:,:]>0
    area=copy.copy(area_old)
    area_sum=np.array([]).reshape(0,area.shape[0],area.shape[1])
    while (area==True).sum() >100:
        start=np.array([np.where(area==True)])[0,:,:]
        reg1=dfs_area(area,(start[0,0],start[1,0]))
        if (reg1==1).sum()>100:
            area_sum=np.concatenate((area_sum,reg1.reshape(1,reg1.shape[0],reg1.shape[1])),axis=0)
        flag=reg1==1
        area[flag]=False
    area_sig=np.repeat(area_sum.reshape(area_sum.shape[0],1,area_sum.shape[1],area_sum.shape[2]),
                       sic.shape[0],axis=(1))
    for jj in np.arange(area_sig.shape[0]):
        for ii in np.arange(sic.shape[0]):
            sic_zw=sic[ii,:,:,:]
            flag=np.nanmin(sic_zw,axis=(0))>sic_max
            area_sig[jj,ii,flag]=0     
    return area_sum,area_sig

def optimize_wave(sic,area,t2m,area_old=np.array([]),
                  sic_old=np.array([]),t2m_old=np.array([]),th=np.array([263,268,272]),
                 overlap_max=0.7,asi='no'):
    '''
    This function checks if main sic decline of the detected warming area is captured in the time period. If not,
    warming wave not considered for automatic detection.
    sic = array of sea ice concentration (dimension = [ALG,Time,X,Y])
    area = mask of detected warm air intrusions for the current time period
    t2m = array of air temperature
    area_old = mask of previously detected waves 
    th = array of temperature thresholds
    overlap_may = maximum of allowed overlap between previous and current detected waves
    asi = if 'yes': use results from ASI algorithm for optimizing wave (usually this function is not needed)
    '''
    dim=sic.ndim
    remove_old=np.array([])    
    if dim<4:
        sic=sic.reshape(1,sic.shape[0],sic.shape[1])
    area_keep=np.array([]).reshape(0,area.shape[1],area.shape[2])
    overlaps=np.zeros((area.shape[0],area.shape[1],area.shape[2]))
    if area_old.shape[0]>0:
        remove=np.zeros((area.shape[0]))
        remove_old=np.zeros((area_old.shape[0]))
        for ii in np.arange(area.shape[0]):
            a2=area[ii,:,:]
            for jj in np.arange(area_old.shape[0]):
                a1=area_old[jj,:,:]
                s2=copy.copy(sic)
                s1=copy.copy(sic_old)
                t2=copy.copy(t2m)
                t1=copy.copy(t2m_old)
                flag=np.broadcast_to(a1==1,t1.shape)
                flag2=np.broadcast_to(a2==1,t2.shape)
                t1[flag!=1]=np.nan
                t2[flag2!=1]=np.nan
                tmean=np.nanmean(t1,axis=(1,2))
                t2mean=np.nanmean(t2,axis=(1,2))
                overlap=np.logical_and(a1==1,a2==1)
                overlap_1=(overlap==1).sum()/(a1==1).sum()
                overlap_2=(overlap==1).sum()/(a2==1).sum()
                if overlap_1<overlap_max and overlap_2<overlap_max: 
                     overlaps[ii,overlap]=1
                else:
                    ratio=(a1==1).sum()/(a2==1).sum()
                    if ratio<0.5:
                        remove_old[jj]=1
        
                    elif ratio>2:
                        remove[ii]=2+jj
                    else:  
                        if asi=='yes':
                            s1_asi=np.repeat(s1[1,:,:,:].reshape(1,s1.shape[1],s1.shape[2],s1.shape[3]),3,axis=0)
                            s2_asi=np.repeat(s2[1,:,:,:].reshape(1,s2.shape[1],s2.shape[2],s2.shape[3]),3,axis=0)  
                            lossx,loss_g,sicl,sicl_g=calc_area_loss(copy.copy(a1),copy.copy(s1_asi),copy.copy(t1))
                            lossx1,loss_g1,sicl1,sicl_g1=calc_area_loss(copy.copy(a2),copy.copy(s2_asi),
                                                                       copy.copy(t2))                            
                        else:    
                            lossx,loss_g,sicl,sicl_g=calc_area_loss(copy.copy(a1),copy.copy(s1),copy.copy(t1))
                            lossx1,loss_g1,sicl1,sicl_g1=calc_area_loss(copy.copy(a2),copy.copy(s2),
                                                                       copy.copy(t2))
                        loss1_sum=np.nanmean(np.nansum(lossx,axis=(0,2)))
                        loss2_sum=np.nanmean(np.nansum(lossx1,axis=(0,2)))
                        loss_ratio=loss1_sum/loss2_sum
                        if loss_ratio>1:
                            remove[ii]=2+jj
                        else:
                            remove_old[jj]=1
                            
        flag=remove>1
        flag2=remove<1
        area_keep=np.delete(area,flag2,axis=0)
        flag3=overlaps>0
        area[flag3]=np.nan            
        area=np.delete(area,flag,axis=0)     
    area_old=copy.copy(area)
    area_old=np.concatenate((area_old,area_keep),axis=0)
    return area,area_old,remove_old


def calc_area_loss(area,sic,t2m,pday=5,th=np.array([263,268,271])):
    '''Calculates area loss and sic drop during warming event
       area = masked array of detected warm air intrusion
       sic = array of sea ice concentration
       pday = days prior to warm air intrusion (should be similar as defined in detect_warming_wave
       
    '''
    if np.ndim(area)<np.ndim(sic):
        area_zw=np.broadcast_to(area,sic.shape)
    sic[area_zw==0]=np.nan  
    loss_out=np.array([]).reshape(0,sic.shape[0],sic.shape[1])
    sic_out=np.array([]).reshape(0,sic.shape[0],sic.shape[1])
    for ii in np.arange(th.shape[0]):
        area_th=copy.copy(area)
        sic_th=copy.copy(sic)
        flag=np.nanmax(t2m,axis=0)>th[ii]
        area_th[flag]=2
        flag=area!=1
        area_th[flag]=0
        if ii<th.shape[0]-1:
            flag=np.nanmax(t2m,axis=0)>=th[ii+1]
            area_th[flag]=0          
        warming_flag=np.broadcast_to(area_th==2,sic_th.shape)
        sic_th[warming_flag!=1]=np.nan     
        warming_flag2=np.broadcast_to(area_th==2,sic_th.shape)     
        area_th[area_th!=2]=np.nan
        area_th[area_th==2]=1
        sic_mean=np.nanmean(sic_th,axis=(2,3))
     
        if np.nanmean(sic_th[:,0:pday,:,:]) > np.nanmean(sic_th[:,-pday:,:,:])-5:
            sic_ave=np.broadcast_to(np.nanmean(sic_th[:,0:pday,:,:],axis=(1,2,3)).reshape(sic_th.shape[0],1),
                                sic_mean.shape)
        else:
            sic_ave=np.broadcast_to(np.nanmean(sic_th[:,-pday:,:,:],axis=(1,2,3)).reshape(sic_th.shape[0],1),
                                sic_mean.shape)          

        sic_loss=sic_ave-sic_mean

        sic_loss[:,0:3]=0
 
        area_loss=(sic_loss/100)*(area_th[:,:]>0).sum()*25*25
        area_loss[:,0:pday]=0
  
        area_loss[area_loss<0]=0
        sic_loss[sic_loss<0]=0
        loss_out=np.concatenate((loss_out,area_loss.reshape(1,area_loss.shape[0],area_loss.shape[1])),axis=0)
        sic_out=np.concatenate((sic_out,sic_loss.reshape(1,sic_loss.shape[0],sic_loss.shape[1])),axis=0)
     
    return loss_out,sic_out

# Supporting functions
def resample_weights(lon_in,lat_in,lon_out,lat_out,radius_of_influence = 25000):
    '''
    calculates the wheights for resampling data to NSIDC 25km polar grid
    lon_in,lat_in = coordinates of input data
    lon_out,lat_out=coordinates of output format
    '''
    swath_def= geometry.SwathDefinition(lons=lon_in, lats=lat_in)
    area_def = geometry.SwathDefinition(lons=lon_out, lats=lat_out)
    valid_input_index, valid_output_index, index_array, distance_array = kd_tree.get_neighbour_info(swath_def
                                                        ,area_def,radius_of_influence=radius_of_influence,
                                                        neighbours=1)
    return valid_input_index, valid_output_index,index_array

def regrid_data(data,lon_in,lat_in,lon_out,lat_out,valid_input_index,valid_output_index,index_array,
                fill_value = np.nan):
    '''
    resamples data to NSIDC 25km polar grid. 
    lon_in,lat_in = coordinates of input data
    lon_out,lat_out=coordinates of output format 
    index parameters obtained from function: resample_weights
    '''
    swath_def_in = geometry.SwathDefinition(lons=lon_in, lats=lat_in)
    swath_def_out = geometry.SwathDefinition(lons=lon_out, lats=lat_out)    
    data_out = kd_tree.get_sample_from_neighbour_info('nn', swath_def_out.shape, data,valid_input_index,
                                                      valid_output_index,index_array,fill_value=fill_value)
    return data_out

def preprocess_data(t2m,osi,asi,cdr,tb6v,tb6h,tb18v,tb18h,tb36v,tb36h,tb89v,tb89h,lat_out,sic_era,rh,nt,bt,limit=75):
    '''
    This function can be used to preprocess data (e.g., apply landmasks/geographic masks)
    '''
    test=copy.copy(t2m)
    flag=np.isnan(nt)
    test[flag]=np.nan
    flag=np.broadcast_to(nt[0,:,:]<65,t2m.shape)
    test[flag]=np.nan
    flag=np.broadcast_to(lat_out<75,test.shape)
    test[flag]=np.nan
    osi[osi<0]=np.nan
    cdr[cdr<0]=np.nan
    osi_old=copy.copy(osi)
    asi_old=copy.copy(asi)
    nt_old=copy.copy(nt)
    osi=iceedge_mask(osi,limit=limit)
    asi=iceedge_mask(asi,limit=limit)
    nt=iceedge_mask(nt,limit=limit)
    tb6v=iceedge_mask(tb6v,limit=limit)
    tb6h=iceedge_mask(tb6h,limit=limit)
    tb18v=iceedge_mask(tb18v,limit=limit)
    tb18h=iceedge_mask(tb18h,limit=limit)
    tb36v=iceedge_mask(tb36v,limit=limit)
    tb36h=iceedge_mask(tb36h,limit=limit)
    tb89v=iceedge_mask(tb89v,limit=limit)
    tb89h=iceedge_mask(tb89h,limit=limit)
    sic_era=iceedge_mask(sic_era,limit=limit)
    rh=iceedge_mask(rh,limit=limit)
    sic=np.concatenate((osi.reshape(1,osi.shape[0],osi.shape[1],osi.shape[2]),asi.reshape(1,
                    asi.shape[0],asi.shape[1],asi.shape[2]),cdr.reshape(1,cdr.shape[0],cdr.shape[1],cdr.shape[2])
                   ,nt.reshape(1,nt.shape[0],nt.shape[1],nt.shape[2])
                   ,bt.reshape(1,bt.shape[0],bt.shape[1],bt.shape[2])
                   ,sic_era.reshape(1,sic_era.shape[0],sic_era.shape[1],sic_era.shape[2])),axis=0)
    tbs=np.concatenate((tb6v.reshape(1,tb6v.shape[0],tb6v.shape[1],tb6v.shape[2]),
                    tb6h.reshape(1,tb6h.shape[0],tb6h.shape[1],tb6h.shape[2]),
                    tb18v.reshape(1,tb18v.shape[0],tb18v.shape[1],tb18v.shape[2]),
                    tb18h.reshape(1,tb18h.shape[0],tb18h.shape[1],tb18h.shape[2]), 
                    tb36v.reshape(1,tb36v.shape[0],tb36v.shape[1],tb36v.shape[2]),
                    tb36h.reshape(1,tb36h.shape[0],tb36h.shape[1],tb36h.shape[2]),
                    tb89v.reshape(1,tb89v.shape[0],tb89v.shape[1],tb89v.shape[2]),
                    tb89h.reshape(1,tb89h.shape[0],tb89h.shape[1],tb89h.shape[2]),                                   
                                ),axis=0)    
    return sic,tbs,test,rh

