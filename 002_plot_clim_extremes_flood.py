#!/usr/bin/env python
# coding: utf-8

# Load packages
from awrams.utils.processing.extract import extract
from awrams.utils.gis import ShapefileDB,extent_from_record
from awrams.utils.extents import get_default_extent
import pandas as pd
import numpy as np
import pickle
import os
from itertools import product
import matplotlib.pyplot as plt

### Set the customer folder and shapefile name
shp_name = 'NRM_clusters'

## Data Request dir folder path
DataRequest_dir = '/g/data/er4/rs2140/data/'
dir_out = '/g/data/er4/zk6340/code/Script_Hydro-projections'

# Which cluster
which_cluster = 'Wet Tropics'

# Which return period
T = 20

# Participating GCMs, Bias correction approaches, Emission scenarios
gcms = ['CNRM-CERFACS-CNRM-CM5','CSIRO-BOM-ACCESS1-0','MIROC-MIROC5','NOAA-GFDL-GFDL-ESM2M']
bias_corr = ['CSIRO-CCAM-r3355-r240x120-ISIMIP2b-AWAP','r240x120-ISIMIP2b-AWAP', 'r240x120-MRNBC-AWAP', 'r240x120-QME-AWAP']
emission = ['rcp45','rcp85']

# Variables
var_name = 'rain_day'
variable = 'pr'

## Read shape file
shp = DataRequest_dir + 'shapefiles/NRM_clusters/' + shp_name + '.shp'
sdb = ShapefileDB(shp)
sdf = sdb.get_records_df()
emap = pickle.load(open(DataRequest_dir + 'shapefiles/NRM_clusters/' + shp_name + '.pkl', 'rb'))

## Which cluster
cluster = {}
for key in emap:
    if which_cluster in key:
        cluster[key] = emap[key]

# Period
period_hist = pd.date_range('1 jan 1976', '31 dec 2005', freq='D')
period_projected = pd.date_range('2080-01-01 12:00:00', '2099-12-31 00:00:00', freq='D')

#Function to get to the path 
def get_path(parameter,which_gcm,which_emission,which_bias_corr):
    return(os.path.join('/g/data/wj02/COMPLIANT/HMINPUT/output/AUS-5/BoM/%s/%s/r1i1p1/%s/latest/day/%s'%(which_gcm,which_emission,which_bias_corr,parameter)))

#Function to get to the file
def get_file(parameter,which_gcm,which_emission,which_bias_corr):
    base_filename = '%s_AUS-5_%s_%s_r1i1p1_%s_v1_day_20060101-20991231.nc'%(parameter,which_gcm,which_emission,which_bias_corr)
    return(base_filename)

#Function to calculate return period: EV1/Gumble distribution
def return_period(data, T):
    xbar=np.mean(data)
    stdev=np.std(data)
    ff=np.sqrt(6)/3.1416*(-0.5772-np.log(-np.log(1-1/T))) 
    qt=xbar+stdev*ff
    return(qt)

# AWAP data
input_path_awap = '/g/data/er4/data/CLIMATE/rain_day/'
pattern = input_path_awap + '%s*.nc' % var_name
ds_awap = extract(input_path_awap, pattern, var_name, cluster, period_hist)
awap_mean = ds_awap.resample(rule='A-DEC').sum().mean()[0]
awap_max = ds_awap.max()[0]
yr_max_awap = ds_awap.resample(rule='A-DEC').max()
awap_return_period = return_period(yr_max_awap, T)[0]

#GCM data
values = [(i, j, k) for i, j, k in product(gcms, bias_corr, emission)]
all_details = []
for val in enumerate(values): 
    input_path_gcm = get_path(variable,val[1][0],val[1][2],val[1][1])
    input_file = get_file(variable,val[1][0],val[1][2],val[1][1])
    pattern_gcm = input_path_gcm + '/' + input_file
    rr_gcm = extract(input_path_gcm, pattern_gcm, variable, cluster, period_projected) # unit in kg m-2 s-1
    ds_gcm = rr_gcm*86400 # unit in mm
    gcm_mean = ds_gcm.resample(rule='A-DEC').sum().mean()[0]
    gcm_max = ds_gcm.max()[0]
    yr_max_gcm = ds_gcm.resample(rule='A-DEC').max()
    gcm_return_period = return_period(yr_max_gcm, T)[0]
    annual_mean_change = ((awap_mean-gcm_mean)/awap_mean)*100
    annual_max_day_change = ((awap_max-gcm_max)/awap_max)*100
    return_period_change = ((awap_return_period-gcm_return_period)/awap_return_period)*100
    dd = {'GCM':val[1][0],'Bias_correction':val[1][2],'Emission':val[1][1],'Annual_mean_prcp_change':annual_mean_change,'Max_1_day_prcp_change':annual_max_day_change,'Return_period_change':return_period_change}
    all_details.append(dd)
all_data = pd.DataFrame(all_details)
print(all_data)
# Write putput to csv
all_data.to_csv(os.path.join(dir_out,'output_file.csv'))

# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# title = "NRM Region: %s"%(which_cluster)
# ax.set_title(title) 
# months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# ax.set_ylabel('Total Rainfall (mm)')
# xx = ax.bar(months, df_m['Wet Tropics'], color = 'skyblue')
# ax.text(x = months, y = df_m['Wet Tropics'],label = df_m['Wet Tropics'], va='center', ha='right',cex = 0.8, col = 'k')
# output_file = 'Rainfall_climatology_%s.jpeg'%(which_cluster)
# plt.savefig(os.path.join('/g/data/er4/zk6340/Hydro_projection',output_file))
# plt.show()