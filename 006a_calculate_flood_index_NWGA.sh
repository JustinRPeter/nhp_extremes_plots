# This script creates one PBS job script for every variable and submits the job.
cd /g/data/er4/zk6340/code/Script_Hydro-projections

for var in "qtot" "rain_day"; do
    # copy the template to a new file
    cp 006a_calculate_flood_index_NWGA_TEMPLATE.pbs PBS_jobs/006a_calculate_flood_index_NWGA_${var}.pbs
    
    wait
    
    # replace var in the job file
    sed -i "s|var|${var}|g" "PBS_jobs/006a_calculate_flood_index_NWGA_${var}.pbs"
            
    wait; sleep 1
    
    # submit job
    qsub PBS_jobs/006a_calculate_flood_index_NWGA_${var}.pbs
    
    wait
done
