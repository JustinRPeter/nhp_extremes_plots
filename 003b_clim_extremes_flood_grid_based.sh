# This script creates one PBS job script for every region and submits the job.

cd /g/data/er4/zk6340/code/Script_Hydro-projections

#for region in 'CS' 'EC' 'MB' 'MN' 'R' 'SS' 'SSWF' 'WT'; do
for region in 'CS' 'EC' 'MB' 'SS' 'SSWF' 'WT'; do

    # copy the template to a new file
    cp 003b_clim_extremes_flood_grid_based_TEMPLATE.pbs PBS_jobs/003b_clim_extremes_flood_grid_based_TEMPLATE_${region}.pbs
    
    wait
    
    # replace region in the job file
    sed -i "s|region|${region}|g" "PBS_jobs/003b_clim_extremes_flood_grid_based_TEMPLATE_${region}.pbs"

    wait; sleep 1
    
    # submit job
    qsub PBS_jobs/003b_clim_extremes_flood_grid_based_TEMPLATE_${region}.pbs
    
    wait
done
