'''
Indicator for change in future flood scenario: catchment aggregated data
Variables: Precipitation/Runoff
Project: CaRSA
#############################################################################################################

Author: Zaved Khan 
Email: zaved.khan@bom.gov.au
18/06/2021

'''
from awrams.utils.processing.extract import extract
from awrams.utils.gis import ShapefileDB,extent_from_record
from awrams.utils.extents import get_default_extent
import pandas as pd
import numpy as np
import pickle
import os, sys
from numpy import f2py # python wrapper for ForTran
import xarray as xr
from itertools import product
from datetime import datetime, timedelta
import argparse
import logging
from scipy.special import gamma, factorial #gamma function
from scipy.stats import skew
from scipy.stats import norm
sys.path.append('/g/data/er4/zk6340/code/hydrodiy')
from hydrodiy.io import iutils
import errno
sys.path.append('/g/data/er4/zk6340/code/Script_CaRSA')
import grFor # generated python wrapper package

# Variable name, catchment
var_name=sys.argv[1] # 'qtot'/'rain_day'
# Which catchment
#selected_catchments = ['Burdekin']
selected_catchments = ['43637150']

# return period
yT = 20

# Future period
#future_period = {'scn1':['2016-01-01 12:00:00','2045-12-31 00:00:00'],'scn2':['2036-01-01 12:00:00','2065-12-31 00:00:00'],'scn3':['2056-01-01 12:00:00','2085-12-31 00:00:00'],'scn4':['2069-12-31 12:00:00','2099-12-31 00:00:00']}
future_period = {'scn1':['2016-01-01 12:00:00','2045-12-31 00:00:00','2030'],'scn3':['2056-01-01 12:00:00','2085-12-31 00:00:00','2070']}

# historical period
period_hist = pd.date_range('1 jan 1976', '31 dec 2005', freq='D')

# Path to historical data
if var_name == 'qtot':
    data_hist = '/g/data/er4/exv563/hydro_projections/data/drought_2018_analysis/awra_v6_static_winds/daily'
else:
    data_hist = '/g/data/er4/data/CLIMATE/rain_day'

# output dir
output_dir='/g/data/er4/zk6340/Hydro_projection'

# Participating GCMs, Bias correction approaches, Emission scenarios
gcms = ['CNRM-CERFACS-CNRM-CM5','CSIRO-BOM-ACCESS1-0','MIROC-MIROC5','NOAA-GFDL-GFDL-ESM2M']
bias_corr = ['CSIRO-CCAM-r3355-r240x120-ISIMIP2b-AWAP','r240x120-ISIMIP2b-AWAP', 'r240x120-MRNBC-AWAP', 'r240x120-QME-AWAP']
emission = ['rcp45','rcp85']

# logger
basename = os.path.basename(os.path.abspath(__file__))
LOGGER = logging.getLogger(basename)
LOGGER.setLevel(logging.INFO)
fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
ft = logging.Formatter(fmt)
flog = os.path.join(output_dir, '{}_{}_{}.log'.format(basename,var_name,selected_catchments[0]))
fh = logging.FileHandler(flog)
fh.setFormatter(ft)
LOGGER.addHandler(fh)

### Set the customer folder and shapefile name
shp_name = 'New England Combined'
## Data Request dir folder path
DataRequest_dir = '/g/data/er4/zk6340/Hydro_projection/%s/'%(shp_name)
## Read shape file
shp = DataRequest_dir + shp_name + '.shp'
sdb = ShapefileDB(shp)
sdf = sdb.get_records_df()

# #Create the extents and save as pkl file if pkl is not available
edef = get_default_extent()
emap = {}
for i,e in enumerate(sdf['HydroID']): 
    print('Dealing with {}'.format(i))
    try:
        emap[e] = extent_from_record(sdb.records[i],edef)
    except Exception as err:
        print('Error in delineation')
pickle.dump(emap, open(DataRequest_dir + shp_name + '.pkl', 'wb'))

## Read the .pkl file
emap = pickle.load(open(DataRequest_dir + shp_name + '.pkl', 'rb'))

#Function to get to the path 
def get_path(parameter,which_gcm,which_emission,which_bias_corr):
    if parameter == 'qtot':
        path = os.path.join('/g/data/wj02/COMPLIANT/HMOUTPUT/output/AUS-5/BoM/AWRALv6-1-%s/%s/r1i1p1/%s/latest/day/%s'%(which_gcm,which_emission,which_bias_corr,parameter))
    else:
        path = os.path.join('/g/data/wj02/COMPLIANT/HMINPUT/output/AUS-5/BoM/%s/%s/r1i1p1/%s/latest/day/%s'%(which_gcm,which_emission,which_bias_corr,parameter))
    return(path)

#Function to get to the file
def get_file(which_gcm,which_bias_corr,which_emission,parameter):
    if parameter == 'qtot':
        base_filename = 'AWRALv6-1-%s_%s_%s_r1i1p1_%s_AUS-5_day_v1_20060101-20991231.nc'%(which_gcm,which_bias_corr,which_emission,parameter)
    else:
        base_filename = '%s_AUS-5_%s_%s_r1i1p1_%s_v1_day_20060101-20991231.nc'%(parameter,which_gcm,which_emission,which_bias_corr)       
    return(base_filename)

#Function to calculate return period: GEV distribution - data requires sorting
def gev_return_period(data, T):
    lamda1= grFor.lh(data,0,len(data))[0]
    lamda2= grFor.lh(data,0,len(data))[1]
    lamda3= grFor.lh(data,0,len(data))[2]
    lamda4= grFor.lh(data,0,len(data))[3]
    c=(2*lamda2/(lamda3+3*lamda2))-(np.log(2)/np.log(3))
    k=7.859*c+2.9554*c**2
    alpha=k*lamda2/(gamma(1+k)*(1-2**(-k)))
    zeta=lamda1-alpha/k*(1-(gamma(1+k)))
    qt=zeta+(alpha/k)*(1-(-np.log(1-1/T))**k)
    return(qt)

# scenarios
values = [(i, j) for i, j in product(selected_catchments,future_period)]

# in future scenario netcdf file, variable 'rain_day' is changed to 'pr'
if var_name == 'rain_day':
    variable = 'pr'
else: 
    variable = 'qtot'  
    
# Generate all the emsemble identities
ensb_mem = [(k, l, m) for k, l, m in product(gcms,emission, bias_corr)]

LOGGER.info('Calculation starts: ') 
# Get the calculation done!
ensb_together = []
for ensb in enumerate(ensb_mem):
    ds_produced = []
    for val in enumerate(values): 
        which_catchment = val[1][0]
        print(which_catchment)
        period_projected = pd.date_range(future_period[val[1][1]][0], future_period[val[1][1]][1], freq='D')
        catchment = {}
        for key in emap:
            if which_catchment in key:
                catchment[key] = emap[key]
        #Annual max for historical data                
        pattern = data_hist + '/%s*.nc' % var_name
        ds_hist = extract(data_hist, pattern, var_name, catchment, period_hist)
        data_mean_hist = ds_hist[which_catchment].values.mean()
        data_max_hist = ds_hist[which_catchment].values.max()
        data_annual_max_hist = ds_hist.resample(rule='A-DEC').max()
        
        #Annual max for future data    
        input_path_gcm = get_path(variable,ensb[1][0],ensb[1][1],ensb[1][2])
        input_file = get_file(ensb[1][0],ensb[1][2],ensb[1][1],variable) 
        pattern_gcm = input_path_gcm + '/' + input_file
        ds_gcm = extract(input_path_gcm, pattern_gcm, variable, catchment, period_projected)
        if var_name == 'rain_day':
            ds_convrt = ds_gcm*86400
        else:
            ds_convrt = ds_gcm
        data_mean_future = ds_convrt[which_catchment].values.mean()
        data_max_future = ds_convrt[which_catchment].values.max()
        data_annual_max_future = ds_convrt.resample(rule='A-DEC').max()
        #Calculate 20 year return period
        hist_yT = gev_return_period(np.sort(data_annual_max_hist[which_catchment].values), yT)
        future_yT = gev_return_period(np.sort(data_annual_max_future[which_catchment].values), yT)
        dd = {'Catchment': which_catchment,'Mean over historical period':data_mean_hist,'Mean over future period':data_mean_future,'Max over historical period':data_max_hist,'Max over future period':data_max_future,'20-yr yT over historical period':hist_yT,'20-yr yT over future period':future_yT,'Future period':future_period[val[1][1]][0],'Scenario':future_period[val[1][1]][2],'GCM':ensb[1][0],'Emission':ensb[1][1],'Bias Correction':ensb[1][2]}
        print (dd)
        ds_produced.append(dd)
    #break
    ensb_together.append(pd.DataFrame(ds_produced)) # Appending pandas dataframes generated in a for loop
LOGGER.info('All calculation completed :): ') 
LOGGER.info('Now we will write output in .csv') 

output_filename= '%s_%s.csv' % (var_name,shp_name)
pd.concat(ensb_together,ignore_index = True).to_csv((os.path.join(output_dir,output_filename)), index=True) # convert from list to pandas dataframe
LOGGER.info('.csv is written') 
LOGGER.info('All completed!') 