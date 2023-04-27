#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import pandas as pd
import numpy as np
import xarray as xr
from itertools import product
from datetime import datetime, timedelta
from scipy.stats import scoreatpercentile # calculate percentile
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches # legend add manually


# In[2]:


dir_in = '/g/data/er4/zk6340/Hydro_projection/data_flood_scenario_qtot/20160101-20451231'
dir_out = '/g/data/er4/zk6340/Hydro_projection/data_flood_scenario_qtot/20160101-20451231'


# In[20]:


zones = [1,2,4,5,6,7,8,9]


# In[21]:


zones[0]


# In[3]:


cluster = 6


# In[4]:


# Which cluster
clusters = {1:'Central Slopes',2:'East Coast',4:'Murray Basin',5:'Monsoonal North',6:'Rangelands',7:'Southern Slopes',8:'SSW Flatlands',9:'Wet Tropics'}
which_cluster = clusters[cluster]


# In[5]:


ds = pd.read_csv(os.path.join(dir_in, 'ds_flood_plot_%s.csv')%(cluster), index_col = 0)


# In[6]:


ds


# In[7]:


#column_names = ['Daily Mean','Daily Max','20-yr Return Period']
column_names = ['Change in Daily Mean','Change in daily Max','Change in Return Period']


# In[8]:


ds_merge = []
for col_name in enumerate(column_names): 
    ds['Percentage Change'] = ds[col_name[1]]
    ds['Indicator'] = col_name[1]
    #ds_one = ds.drop(['Daily Mean','Daily Max','20-yr Return Period'], axis=1)
    ds_one = ds.drop(['Change in Daily Mean','Change in daily Max','Change in Return Period'], axis=1)
    ds_merge.append(ds_one)
ds_complete = pd.concat([ds_merge[0],ds_merge[1],ds_merge[2]])


# In[9]:


ds_complete.loc[ds_complete['Indicator'] == 'Change in Daily Mean', 'Indicator'] = 'Daily Mean'
ds_complete.loc[ds_complete['Indicator'] == 'Change in daily Max', 'Indicator'] = 'Daily Max'
ds_complete.loc[ds_complete['Indicator'] == 'Change in Return Period', 'Indicator'] = 'Return Period'


# In[10]:


def ds_plot(which_indicator,which_rcp):
    ds_indicator = ds_complete[ds_complete['Indicator']==which_indicator]
    what_change = ds_indicator[ds_indicator['Emission']==which_rcp]['Percentage Change']
    return(what_change)


# In[11]:


values_plot = ds_plot('Daily Mean','rcp45')
values_plot


# In[12]:


np.percentile(values_plot, 10)


# In[13]:


np.percentile(values_plot, 90)


# In[14]:


np.percentile(values_plot, 50)


# In[15]:


# fig, ax = plt.subplots()
# fig.set_size_inches((15, 10.5))
# sns.boxplot(x="Indicator", y="Percentage Change", hue="Emission", data=ds_complete, palette="colorblind", ax=ax, width = 0.3)
# sns.swarmplot(x="Indicator", y="Percentage Change", hue="Emission", data=ds_complete, palette="colorblind", ax=ax)
# plt.axhline(y=0, color='k', linestyle='--')
# plt.ylabel("% Change in 2016-2045 relative to 1976-2005", size=16)
# plt.xlabel("",size=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# #plt.title("Projected Flood Scenario over %s"%(which_cluster), size=18)
# # output_file = 'Flood_plot_%s.jpeg'%(which_cluster) 
# # plt.savefig(os.path.join(dir_out,output_file))


# In[16]:


#Custom percentile
#https://stackoverflow.com/questions/8681199/python-matplotlib-boxplot-how-to-show-percentiles-0-10-25-50-75-90-and-100


# In[17]:


perc = [scoreatpercentile(ds_plot('Daily Mean','rcp45'),10), scoreatpercentile(ds_plot('Daily Mean','rcp45'),50)
        ,scoreatpercentile(ds_plot('Daily Mean','rcp45'),50), scoreatpercentile(ds_plot('Daily Mean','rcp45'),90)
        ,scoreatpercentile(ds_plot('Daily Mean','rcp85'),10), scoreatpercentile(ds_plot('Daily Mean','rcp85'),50)
        ,scoreatpercentile(ds_plot('Daily Mean','rcp85'),50), scoreatpercentile(ds_plot('Daily Mean','rcp85'),90)
        ,scoreatpercentile(ds_plot('Daily Max','rcp45'),10),scoreatpercentile(ds_plot('Daily Max','rcp45'),50)
        ,scoreatpercentile(ds_plot('Daily Max','rcp45'),50),scoreatpercentile(ds_plot('Daily Max','rcp45'),90)
        ,scoreatpercentile(ds_plot('Daily Max','rcp85'),10),scoreatpercentile(ds_plot('Daily Max','rcp85'),50)
        ,scoreatpercentile(ds_plot('Daily Max','rcp85'),50),scoreatpercentile(ds_plot('Daily Max','rcp85'),90)
        ,scoreatpercentile(ds_plot('Return Period','rcp45'),10),scoreatpercentile(ds_plot('Return Period','rcp45'),50)
        ,scoreatpercentile(ds_plot('Return Period','rcp45'),50),scoreatpercentile(ds_plot('Return Period','rcp45'),90)
        ,scoreatpercentile(ds_plot('Return Period','rcp85'),10),scoreatpercentile(ds_plot('Return Period','rcp85'),50)
        ,scoreatpercentile(ds_plot('Return Period','rcp85'),50),scoreatpercentile(ds_plot('Return Period','rcp85'),90)]


# In[18]:


perc


# In[19]:


fig, ax = plt.subplots()
fig.set_size_inches((15, 10.5))
ax.broken_barh([(1,0.2)], (perc[0], perc[1]-perc[0]),color ='steelblue',edgecolor = 'k')
ax.broken_barh([(1,0.2)], (perc[2], perc[3]-perc[2]),color ='steelblue',edgecolor = 'k')
ax.broken_barh([(1.3,0.2)], (perc[4], perc[5]-perc[4]),color ='darkgoldenrod',edgecolor = 'k')
ax.broken_barh([(1.3,0.2)], (perc[6], perc[7]-perc[6]),color ='darkgoldenrod',edgecolor = 'k')
ax.broken_barh([(4,0.2)], (perc[8], perc[9]-perc[8]),color ='steelblue',edgecolor = 'k')
ax.broken_barh([(4,0.2)], (perc[10], perc[11]-perc[10]),color ='steelblue',edgecolor = 'k')
ax.broken_barh([(4.3,0.2)], (perc[12], perc[13]-perc[12]),color ='darkgoldenrod',edgecolor = 'k')
ax.broken_barh([(4.3,0.2)], (perc[14], perc[15]-perc[14]),color ='darkgoldenrod',edgecolor = 'k')
ax.broken_barh([(7,0.2)], (perc[16], perc[17]-perc[16]),color ='steelblue',edgecolor = 'k')
ax.broken_barh([(7,0.2)], (perc[18], perc[19]-perc[18]),color ='steelblue',edgecolor = 'k')
ax.broken_barh([(7.3,0.2)], (perc[20], perc[21]-perc[20]),color ='darkgoldenrod',edgecolor = 'k')
ax.broken_barh([(7.3,0.2)], (perc[22], perc[23]-perc[22]),color ='darkgoldenrod',edgecolor = 'k')
ax.set_xticklabels([])
rcp45 = mpatches.Patch(color='steelblue', label='rcp45')
rcp85 = mpatches.Patch(color='darkgoldenrod', label='rcp85')
plt.legend(handles=[rcp45,rcp85], bbox_to_anchor=(1.01, 1), loc='upper left')
#plt.legend(loc="upper left")
plt.axhline(y=0, color='k', linestyle='--')
plt.ylabel("% Change in 2016-2045 relative to 1976-2005", size=16)
plt.xlabel("",size=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Adding label
x = np.array([1.25,4.25,7.25])
my_xticks = ['Daily Mean','Daily Max','Return Period']
plt.xticks(x, my_xticks)
#plt.show()
plt.title("Projected Flood Scenario over %s"%(which_cluster), size=18)
output_file = 'Flood_plot_%s.jpeg'%(which_cluster) 
plt.savefig(os.path.join(dir_out,output_file))


# In[ ]:




