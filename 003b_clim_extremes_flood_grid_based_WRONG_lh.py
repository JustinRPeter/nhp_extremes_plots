'''
Indicator for change in future flood scenario:
This script calculates percentage change in mean, max, return period for each of the eight NRM clusters.
-Base of comparison: Historical GCM data period over 1976-2005
-Future Scenario GCM data period: 2016-2045, 2069-2099
-Return period is calculated using GEV distribution

Variables: Precipitation/Runoff
Project: Hydro-projection
#############################################################################################################

Author: Zaved Khan 
Email: zaved.khan@bom.gov.au
14/12/2020

'''
import os, sys
import pandas as pd
import numpy as np
import xarray as xr
from itertools import product
from datetime import datetime, timedelta
import argparse
import logging
from scipy.special import gamma, factorial #gamma function
sys.path.append('/g/data/er4/zk6340/code/hydrodiy')
from hydrodiy.io import iutils
import errno

# parameter
parameter = 'qtot'

# return period
yT = 20

# projection period
yr_st = '20160101'
yr_end = '20451231'

## dir path
dir_in = '/g/data/er4/zk6340/Hydro_projection/data_flood_scenario_%s'%(parameter)
dir_out = '/g/data/er4/zk6340/Hydro_projection/data_flood_scenario_%s/%s-%s'%(parameter,yr_st,yr_end)

NRM_area = sys.argv[1]

# Which cluster
#clusters = {'CS':1,'EC':2,'MB':4,'MN':5,'R':6,'SS':7,'SSWF':8,'WT':9} 
clusters = {'CS':1,'EC':2,'MB':4,'SS':7,'SSWF':8,'WT':9} #Split the job for Rangelands, Monsoonal North
which_cluster = clusters[NRM_area]

# historical period
hist_st = '19760101'
hist_end = '20051231'

# Participating GCMs, Bias correction approaches, Emission scenarios
gcms = ['CNRM-CERFACS-CNRM-CM5','CSIRO-BOM-ACCESS1-0','MIROC-MIROC5','NOAA-GFDL-GFDL-ESM2M']
bias_corr = ['CSIRO-CCAM-r3355-r240x120-ISIMIP2b-AWAP','r240x120-ISIMIP2b-AWAP', 'r240x120-MRNBC-AWAP', 'r240x120-QME-AWAP']
emission = ['rcp45','rcp85']

#Function to get to the file
def get_file(parameter,which_gcm,which_emission,which_bias_corr, which_metric, yr_end, yr_st):
    base_filename = '%s_AUS-5_%s_%s_r1i1p1_%s_%s_%s-%s.nc'%(parameter,which_gcm,which_emission,which_bias_corr,which_metric,yr_end,yr_st)
    return(base_filename)

#Function to get percentage change for 'Mean' and 'Max'
def percentage_change(parameter, gcm, scenario, bias_correction, statistics, hist_end, hist_st, yr_end, yr_st):
    # Historical period
    filename_hist = get_file(parameter, gcm,'historical', bias_correction, statistics, hist_end, hist_st)
    ds_hist = xr.open_dataset(os.path.join(dir_in,filename_hist)) 
    ds_hist_cluster = ds_hist[parameter].where(mask == which_cluster)
    ds_hist_cluster_spatial_mean = ds_hist_cluster.NRM_cluster.mean().item(0)
    # Future scenario
    filename_scenario = get_file(parameter, gcm, scenario, bias_correction, statistics,yr_end,yr_st)
    ds_scenario = xr.open_dataset(os.path.join(dir_in,filename_scenario)) 
    ds_scenario_cluster = ds_scenario[parameter].where(mask == which_cluster)
    ds_scenario_cluster_spatial_mean = ds_scenario_cluster.NRM_cluster.mean().item(0)   
    percent_change = ((ds_scenario_cluster_spatial_mean-ds_hist_cluster_spatial_mean)*100)/ds_hist_cluster_spatial_mean  
    return(percent_change)

# Calculates the L/LH moments: 
# Note: L-moments when h = 0 
def lh(x, h):
    lh1 = 0
    lh2 = 0
    lh3 = 0
    lh4 = 0
    ts = np.sort(x)
    n = len(ts)
    for i in range (0, len(ts)): 
        cl0 = 1
        if h>0:
            for j in range (0, h): 
                cl0 = cl0*(i-j)/(j+1)
        cl1 = cl0*(i+1-h-1)/(h+1)
        cl2 = cl1*(i+1-h-2)/(h+2)
        cl3 = cl2*(i+1-h-3)/(h+3)
        cr1 = n-(i+1)
        cr2 = cr1*(n-(i+1)-1)/2
        cr3 = cr2*(n-(i+1)-2)/3
        lh1 = lh1+cl0* ts[i]
        lh2 = lh2+(cl1-cl0*cr1)* ts[i]
        lh3 = lh3+(cl2-2*cl1*cr1+cl0*cr2)* ts[i]
        lh4 = lh4+(cl3-3*cl2*cr1+3*cl1*cr2-cl0*cr3)* ts[i] 
    c0 = 1
    if h>0:
        for j in range (0, h): 
            c0 = c0*(n+1-(j+1))/(j+1)
    c1 = c0*(n+1-h-1)/(h+1)
    c2 = c1*(n+1-h-2)/(h+2)
    c3 = c2*(n+1-h-3)/(h+3)
    c4 = c3*(n+1-h-4)/(h+4)
    lh1 = lh1/c1
    lh2 = lh2/c2/2
    lh3 = lh3/c3/3
    lh4 = lh4/c4/4
    return(lh1,lh2,lh3,lh4)

#Function to calculate return period: GEV distribution
def gev_return_period(data, T):
    lamda1= lh(data, 0)[0]
    lamda2= lh(data, 0)[1]
    lamda3= lh(data, 0)[2]
    lamda4= lh(data, 0)[3]
    c=(2*lamda2/(lamda3+3*lamda2))-(np.log(2)/np.log(3))
    k=7.859*c+2.9554*c**2
    alpha=k*lamda2/(gamma(1+k)*(1-2**(-k)))
    zeta=lamda1-alpha/k*(1-(gamma(1+k)))
    qt=zeta+(alpha/k)*(1-(-np.log(1-1/T))**k)
    return(qt)

# #Function to calculate return period: LP3 distribution
# def lp3_return_period(data, T):
#     xbar=np.mean(data)
#     stdev=np.std(data)
#     xskew=skew(data)
#     zp=norm.ppf(1-1/T)
#     ff=(2/xskew)*((1+(zp*xskew)/6-xskew**2/36)**3)-2/xskew
#     qt=xbar+stdev*ff
#     return(qt)

# #Function to calculate return period: EV1/Gumble distribution
# def ev1_return_period(data, T):
#     xbar=np.mean(data)
#     stdev=np.std(data)
#     ff=np.sqrt(6)/3.1416*(-0.5772-np.log(-np.log(1-1/T))) 
#     qt=xbar+stdev*ff
#     return(qt)

#Function to get annual maximum timeseries for a grid cell
def ts_data(dataset, lat, lon, parameter):
    ds = dataset.sel(lat = lat, lon = lon, method = 'nearest')[parameter].values
    return(ds)

# logger
basename = os.path.basename(os.path.abspath(__file__))
LOGGER = logging.getLogger(basename)
LOGGER.setLevel(logging.INFO)
fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
ft = logging.Formatter(fmt)
flog = os.path.join(dir_out, '{}_{}.log'.format(basename,which_cluster))
fh = logging.FileHandler(flog)
fh.setFormatter(ft)
LOGGER.addHandler(fh)

# Read cluster mask
mask = xr.open_dataset(os.path.join(dir_in,'NRM_clusters.nc'))

# picking up all the grid cells for a particular cluster
grid_cells = mask.where(mask == which_cluster).NRM_cluster.to_dataframe().reset_index()
grid_cells = grid_cells.dropna(axis=0).reset_index()

# Generate all the emsemble identities
values = [(i, k, x) for i, k, x in product(gcms,emission, bias_corr)]

LOGGER.info('Calculation loop starts: ') 
# Calculate : Percentage change in Mean, Max, Return period
cluster_flood_indicator=[]
for val in enumerate(values):   
#for val in enumerate(values[0]):  
    #Change in mean
    change_daily_mean = percentage_change(parameter, val[1][0], val[1][1], val[1][2], 'Mean', hist_end, hist_st, yr_end, yr_st)
    #Change in max
    change_daily_max = percentage_change(parameter, val[1][0], val[1][1], val[1][2], 'Max', hist_end, hist_st, yr_end, yr_st)    
    #Change in return period
    # Annual Max - for historical period
    filename_hist_annual_max = get_file(parameter,val[1][0],'historical',val[1][2],'Annual_Max',hist_end,hist_st)
    ds_hist_annual_max = xr.open_dataset(os.path.join(dir_in,filename_hist_annual_max))
    # Annual Max - for scenario period
    filename_scenario_annual_max = get_file(parameter,val[1][0],val[1][1],val[1][2],'Annual_Max',yr_end,yr_st)
    ds_scenario_annual_max = xr.open_dataset(os.path.join(dir_in,filename_scenario_annual_max))  
    # calculate retrun period for the historical period
    all_yT = []
    for i in range (0, len(grid_cells)): 
        hist_yT_point = gev_return_period(ts_data(ds_hist_annual_max, grid_cells.lat[i], grid_cells.lon[i], parameter), yT)
        scenario_yT_point = gev_return_period(ts_data(ds_scenario_annual_max, grid_cells.lat[i], grid_cells.lon[i], parameter), yT)       
        yT_point = {'hist_yT':hist_yT_point,'future_yT':scenario_yT_point}
        all_yT.append(yT_point)
    all_return_period = pd.DataFrame(all_yT)   
    change_return_period = ((all_return_period['future_yT'].mean()-all_return_period['hist_yT'].mean())*100)/all_return_period['hist_yT'].mean()   
    dd = {'GCM':val[1][0],'Bias Correction':val[1][2],'Emission':val[1][1],'Change in Daily Mean':change_daily_mean,'Change in daily Max':change_daily_max, 'Change in Return Period':change_return_period}
    cluster_flood_indicator.append(dd)
    LOGGER.info('All done - Ensemble:%s - %s - %s'%(val[1][0],val[1][2],val[1][1])) 
ds_flood_plot = pd.DataFrame(cluster_flood_indicator)
LOGGER.info('Calculation loop has completed :): ') 
LOGGER.info('Now we will write output in .csv') 

# Writing output in csv
output_file = 'ds_flood_plot_%s.csv' % (which_cluster)
ds_flood_plot.to_csv((os.path.join(dir_out,output_file)), index=True)
LOGGER.info('.csv is written') 
LOGGER.info('All completed!') 