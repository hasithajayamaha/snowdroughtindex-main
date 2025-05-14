.. _arc_cluster:

====================================
ARC Cluster Environment (UCalgary HPC)
====================================

This document provides a reference for the ARC (Advanced Research Computing) Cluster at the University of Calgary, which will be used for running Snow Drought Index workflows and other large-scale computations.

Overview
--------
- **ARC** is a high-performance computing (HPC) cluster for research at UCalgary.
- Composed of hundreds of interconnected servers, including large-memory and GPU nodes.
- Supports serial, multi-threaded (OpenMP), distributed (MPI), and GPU-accelerated jobs.

Access & Usage
--------------
- **Account Required:** UCalgary IT account (external collaborators need a UCalgary project leader).
- **Login:** SSH to `arc.ucalgary.ca` (port 22) from campus or via UCalgary VPN.
- **Job Scheduler:** SLURM (`salloc` for interactive, `sbatch` for batch jobs).

Storage
-------
- `/home`: User home directories.
- `/scratch`: Temporary, high-capacity storage for large jobs.
- `/work`: Persistent project storage.
- `/tmp`, `/var/tmp`, `/dev/shm`, `/run/user/$uid`: Temporary/in-memory storage.

Job Submission Examples
-----------------------
- **Interactive CPU job:**
  .. code-block:: bash

     salloc --mem=1G -c 1 -N 1 -n 1 -t 01:00:00

- **Interactive GPU job:**
  .. code-block:: bash

     salloc --mem=1G -t 01:00:00 -p gpu-v100 --gres=gpu:1

- **Batch job:**
  .. code-block:: bash

     sbatch job-script.slurm

Partitions & Hardware
---------------------
- Use `#SBATCH --partition=<partition_name>` in job scripts to select hardware pools (e.g., `gpu-v100`, `bigmem`, `cpu2013`, `lattice`, `parallel`, `single`).
- Set job time limits with `#SBATCH --time=hh:mm:ss` (e.g., up to 7 days for `single` partition).

Software & Modules
------------------
- Load software environments with the `module` command.
- Common tools: Python, R, MATLAB, TensorFlow, PyTorch, and more.

Best Practices
--------------
- Do not run heavy computations on the login node.
- Release resources after use (`exit` from interactive sessions).
- Use `/scratch` for large, temporary data.
- Monitor jobs and partitions with `sinfo` and `scontrol show partitions`.

Support
-------
- For help: support@hpc.ucalgary.ca

Reference
---------
- [ARC Cluster Guide - University of Calgary](https://rcs.ucalgary.ca/ARC_Cluster_Guide#Hardware)

.. note::
   This document is for internal reference. Update as needed if ARC policies or hardware change, or if workflow integration requirements evolve.

.. note::
   For distributed Python jobs, use MPI with the mpi4py library. Dask is not officially supported on ARC. See the example below for running parallel Python jobs with MPI/mpi4py.

Distributed Python with MPI/mpi4py
----------------------------------
- Use the `mpi4py` library for distributed parallelism in Python.
- Example Python code:

  .. code-block:: python

     from mpi4py import MPI
     import numpy as np

     comm = MPI.COMM_WORLD
     rank = comm.Get_rank()
     size = comm.Get_size()

     # Example: split data among ranks
     data = None
     if rank == 0:
         data = np.arange(100, dtype='i')
         chunks = np.array_split(data, size)
     else:
         chunks = None
     local_data = comm.scatter(chunks, root=0)

     # Each process does work on its chunk
     local_result = np.sum(local_data)

     # Gather results at root
     results = comm.gather(local_result, root=0)
     if rank == 0:
         print('Sum from all ranks:', results)

- Example SLURM batch script for launching an MPI Python job:

  .. code-block:: bash

     #!/bin/bash
     #SBATCH --job-name=mpi-snow
     #SBATCH --partition=single
     #SBATCH --nodes=2
     #SBATCH --ntasks-per-node=4
     #SBATCH --mem=16G
     #SBATCH --time=02:00:00
     #SBATCH --output=mpi-snow-%j.out

     module load python/3.9 openmpi
     source ~/myenv/bin/activate  # or load your environment
     srun python your_mpi_script.py 