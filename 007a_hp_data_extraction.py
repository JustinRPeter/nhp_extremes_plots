'''
Data extraction for HP project
Variables: currently 'Runoff' only (code needs to be generic for other variables)
Project: HP
#############################################################################################################

Author: Zaved Khan 
Email: zaved.khan@bom.gov.au
08/07/2021

'''
from awrams.utils.processing.extract import extract
from awrams.utils.gis import ShapefileDB,extent_from_record
from awrams.utils.extents import get_default_extent
import os, sys
import pandas as pd
import pickle
from itertools import product
from datetime import datetime, timedelta
import argparse
import logging
import errno
sys.path.append('/g/data/er4/zk6340/code/hydrodiy')
from hydrodiy.io import iutils

# variable name
parameter = 'qtot'

# Extraction period
#period = {'hist':['1985-01-01 12:00:00','2005-12-31 12:00:00']}
period = {'scn1':['2031-01-01 12:00:00','2050-12-31 12:00:00'],'scn2':['2051-01-01 12:00:00','2070-12-31 12:00:00']}

# Scenario
#emission = ['historical']
emission = ['rcp85'] #'rcp45'

## output folder path
customer_dir = '/g/data/er4/zk6340/Hydro_projection/data_extraction'

# logger
basename = os.path.basename(os.path.abspath(__file__))
LOGGER = logging.getLogger(basename)
LOGGER.setLevel(logging.INFO)
fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
ft = logging.Formatter(fmt)
flog = os.path.join(customer_dir, '{}_{}_{}.log'.format(basename,parameter,emission[0]))
fh = logging.FileHandler(flog)
fh.setFormatter(ft)
LOGGER.addHandler(fh)

### Set the customer folder and shapefile name
shp_name = '35catchments'

## Data Request dir folder path
DataRequest_dir = '/g/data/er4/zk6340/Hydro_projection/%s/'%(shp_name)

## Read shape file
shp = DataRequest_dir + shp_name + '.shp'
sdb = ShapefileDB(shp)
sdf = sdb.get_records_df()

# #Create the extents and save as pkl file 
# edef = get_default_extent()
# emap = {}
# # for i,e in enumerate(sdf['name']): 
# for i,e in enumerate(sdf['SHAPE_LEN']): 
# #    emap[int(e)] = extent_from_record(sdb.records[i],edef)
#     emap[e] = extent_from_record(sdb.records[i],edef)
# pickle.dump(emap, open(DataRequest_dir + 'shapefiles/WA/' + shp_name + '.pkl', 'wb'))

### Source the existing extents file for the above shape file
emap = pickle.load(open(DataRequest_dir + shp_name + '.pkl', 'rb'))

# Participating GCMs, Bias correction approaches
gcms = ['CNRM-CERFACS-CNRM-CM5','CSIRO-BOM-ACCESS1-0','MIROC-MIROC5','NOAA-GFDL-GFDL-ESM2M']
bias_corr = ['CSIRO-CCAM-r3355-r240x120-ISIMIP2b-AWAP','r240x120-ISIMIP2b-AWAP', 'r240x120-MRNBC-AWAP', 'r240x120-QME-AWAP']

# Generate all the emsemble identities
values = [(i, k, x) for i, k, x in product(gcms, emission, bias_corr)]

#Function to get to the path 
def get_path(parameter,which_gcm,which_emission,which_bias_corr):
    if parameter == 'qtot':
        path = os.path.join('/g/data/wj02/COMPLIANT/HMOUTPUT/output/AUS-5/BoM/AWRALv6-1-%s/%s/r1i1p1/%s/latest/day/%s/'%(which_gcm,which_emission,which_bias_corr,parameter))
    else:
        path = os.path.join('/g/data/wj02/COMPLIANT/HMINPUT/output/AUS-5/BoM/%s/%s/r1i1p1/%s/latest/day/%s/'%(which_gcm,which_emission,which_bias_corr,parameter))
    return(path)

LOGGER.info('Calculation starts: ') 
# Get the calculation done!
for val in enumerate(values):    
    for key, value in period.items():
                LOGGER.info('Dealing with:%s - %s - %s'%(val[1][0],val[1][1],val[1][2]))
                v = parameter
                input_path = get_path(parameter,val[1][0],val[1][1],val[1][2])
                pattern = input_path + '*%s*.nc' % v                    
                extraction_period = pd.date_range(value[0], value[1], freq='D')
                LOGGER.info('input file path:%s'%(pattern))
                ddf = extract(input_path, pattern, v, emap, extraction_period)
                # save daily/monthly/annual data in mm
                LOGGER.info('Now we will write output in .csv')
                dir_name = '%s_VIC_%s_%s-%s'%(v,val[1][1],extraction_period[-1].strftime('%Y%m%d'),extraction_period[0].strftime('%Y%m%d'))
                output_path = os.path.join(customer_dir,dir_name)
                os.makedirs(output_path, exist_ok=True)
                filename_daily = '%s_%s_%s_%s_daily.csv'%(v,shp_name,val[1][2],val[1][0])
                filename_monthly = '%s_%s_%s_%s_monthly.csv'%(v,shp_name,val[1][2],val[1][0])
                filename_annual = '%s_%s_%s_%s_annual.csv'%(v,shp_name,val[1][2],val[1][0])
                ddf.to_csv((os.path.join(output_path,filename_daily)), index=True)  
                ddf.resample('MS').sum().to_csv((os.path.join(output_path,filename_monthly)), index=True) 
                ddf.resample(rule='A-DEC').sum().to_csv((os.path.join(output_path,filename_annual)), index=True)
                LOGGER.info('Completed for:%s - %s - %s -%s-%s'%(val[1][0],val[1][1],val[1][2],extraction_period[-1].strftime('%Y%m%d'),extraction_period[0].strftime('%Y%m%d')))    
                #break            
LOGGER.info('All completed!')