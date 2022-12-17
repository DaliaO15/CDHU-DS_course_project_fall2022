# Instructions Chalmers Cluster

## Connect with ssh
```bash
ssh -L 8080:localhost:6006 username@alvis1.c3se.chalmers.se   
```

## Load modules
```bash
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1 matplotlib/3.4.3-foss-2021b scikit-learn/1.0.1-foss-2021b JupyterLab/3.1.6-GCCcore-11.2.0
pip install split-folders
```

## Git clone project with accesstoken (and checkout branch with code)

```bash
git clone https://github.com/DaliaO15/CDHU-DS_course_project_fall2022.git
```

## Use interactive mode

Example, 4h timeout, 2xA100 GPU:
```bash
srun -A SNIC2022-22-1091 -p alvis -t 4:00:00 --gpus-per-node=A100:2 jupyter notebook
```
You can find the link to access the jupyter environment in the command line.

## Shared files
You can find our shared files under: 
```bash
/mimer/NOBACKUP/groups/snic2022-22-1091/museumFaces
/mimer/NOBACKUP/groups/snic2022-22-1091/FairFace
```

## Use sbatch to run notebook

Create shell script, please don't delete the timeout (-t):

```bash
#!/usr/bin/env bash
#SBATCH -A SNIC2022-22-1091 -p alvis
#SBATCH -t 1:00:00
#SBATCH --gpus-per-node=A100:2
#SBATCH --nodes 1

module purge
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1 matplotlib/3.4.3-foss-2021b scikit-learn/1.0.1-foss-2021b JupyterLab/3.1.6-GCCcore-11.2.0pip install split_folders
pip install split-folders

jupyter lab
```

sbatch will create two log files in the directory you started the job. In the slurm file you can find the link to access the jupyter environment.

## When you are done

Make sure that you have no running tasks:
```bash
jobinfo -u username
```

Check remaining GPU time for project:
```bash
projinfo
```

# Face extraction.

Use the following commands to install dlib with CUDA support:
```bash

!git clone https://github.com/davisking/dlib.git
!cd dlib
!mkdir build
!cd build\n
!cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
!cmake --build .
!cd ..
!python setup.py install --set USE_AVX_INSTRUCTIONS=1 --set DLIB_USE_CUDA=1

```

Note: Make sure to install Visula studio (For Windows), cmake and CUDA drivers before running the above

The following command will run the face extraction module:
```
python3 basic_fd_cropping_CNN.py -sf input_folder -ef output_folder 
```
Example directory to run command in:
```
project    
│
└───input_folder
│   │ image_with_faces_1.jpg
│   │ image_with_faces_1.jpg
│   │ ...
│
└───output_folder
    │
```


