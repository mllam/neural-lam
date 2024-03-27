#!/bin/bash -l
#SBATCH --job-name=dask-job
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=postproc
#SBATCH --account=s83
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=444G
#SBATCH --time=5-00:00:00
#SBATCH --exclusive

# Activate Python environment
conda activate neural-lam

# Set up environment variables
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=600s
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=600s

# Launch the SLURMCluster
srun python -c "
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

JOBS = 2
CORES = 256
PROCESSES = 2
workers = JOBS * PROCESSES

cluster = SLURMCluster(
    queue='postproc',
    account='s83',
    processes=PROCESSES,
    cores=CORES,
    memory='444GB',
    local_directory='/scratch/mch/sadamov/temp',
    shared_temp_directory='/scratch/mch/sadamov/temp',
    log_directory='lightning_logs',
    shebang='#!/bin/bash',
    interface='hsn0',
    walltime='5-00:00:00',
    job_extra_directives=['--exclusive'],
)
cluster.scale(jobs=JOBS)
client = Client(cluster)
client.wait_for_workers(workers)
"

# Run the main Python script
srun python create_zarr_archive.py --data_in /scratch/mch/dealmeih/kenda/ \
                                   --data_out /scratch/mch/sadamov/data.zarr \
                                   --indexpath /scratch/mch/sadamov/temp

# Close the client
srun python -c "client.close()"
