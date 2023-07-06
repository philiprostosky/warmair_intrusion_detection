#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warmair_detection_algorithm as wd
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import copy
import matplotlib.pyplot as plt
# example detection of a warming wave in April 2020
#load data: sea ice concentration and temperature observations from 01.04.2020 - 01.05.2020 
t2m=np.load('/home/prostosky/postdoc/warmair_climatology/data/t2m_example.npy')
osi=np.load('/home/prostosky/postdoc/warmair_climatology/data/osi_example.npy')
asi=np.load('/home/prostosky/postdoc/warmair_climatology/data/asi_example.npy')
cdr=np.load('/home/prostosky/postdoc/warmair_climatology/data/cdr_example.npy')
nt=np.load('/home/prostosky/postdoc/warmair_climatology/data/nt_example.npy')
bt=np.load('/home/prostosky/postdoc/warmair_climatology/data/bt_example.npy')
tb6v=np.load('/home/prostosky/postdoc/warmair_climatology/data/tb6v_example.npy')
tb6h=np.load('/home/prostosky/postdoc/warmair_climatology/data/tb6h_example.npy')
tb18v=np.load('/home/prostosky/postdoc/warmair_climatology/data/tb18v_example.npy')
tb18h=np.load('/home/prostosky/postdoc/warmair_climatology/data/tb18h_example.npy')
tb36v=np.load('/home/prostosky/postdoc/warmair_climatology/data/tb36v_example.npy')
tb36h=np.load('/home/prostosky/postdoc/warmair_climatology/data/tb36h_example.npy')
tb89v=np.load('/home/prostosky/postdoc/warmair_climatology/data/tb89v_example.npy')
tb89h=np.load('/home/prostosky/postdoc/warmair_climatology/data/tb89h_example.npy')
coord=nc.Dataset('/home/prostosky/postdoc/warmair_climatology/data/coords.nc','r')
lat_out=coord.variables['lat'][:,:]
era_sic=np.load('/home/prostosky/postdoc/warmair_climatology/data/era_sic_example.npy')
era_rh=np.load('/home/prostosky/postdoc/warmair_climatology/data/era_rh_example.npy')
landmask=np.load('/home/prostosky/postdoc/warmair_climatology/data/landmask.npy')
th=np.array([263,268,271]) # warming category thresholds



sic,tbs,t2m_p,rh=wd.preprocess_data(t2m,osi,asi,cdr,tb6v,tb6h,tb18v,tb18h,tb36v,tb36h,tb89v,tb89h,lat_out,era_sic,era_rh,nt,bt,limit=65)  #applying ice-egde mask            
area_warming,area_sic=wd.detect_warming_wave(copy.copy(t2m_p),copy.copy(sic),th,landmask=landmask,ende=-10)

warming_categories=area_warming[0,:,:]
get_ipython().run_line_magic('matplotlib', 'widget')
warming_flag=np.broadcast_to(warming_categories[:,:]==1,t2m.shape)
t2m[warming_flag==0]=np.nan    
if th.shape[0]>0:
    for ii in np.arange(th.shape[0]):
        flag=np.nanmax(t2m[0:-5,:,:],axis=0)>th[ii]
        warming_categories[flag]=ii+1
plt.imshow(warming_categories)      


# In[11]:





# In[ ]:




