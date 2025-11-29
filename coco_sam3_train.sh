#!/bin/bash
#SBATCH --partition amdgpulong
#SBATCH --exclude=g11,g12
#SBATCH --nodes 1                   # Number of compute nodes (= servers)
#SBATCH --ntasks-per-node 4         # Number of processes per node (should correspond to the number of GPUs)
#SBATCH --cpus-per-task 10          # Number of CPU cores per process (related to num_workers in config)
#SBATCH --gres gpu:4                # Number of GPUs per node
##BATCH --mem-per-gpu 250G          # Total memory for all nodes
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user kolomcon@student.cvut.cz
#SBATCH --output /home/kolomcon/logs/sam2.1/sam_train1_coco_%J.log


##SBATCH --dependency=afterok:10101341
##SBATCH --time 0-00:01:00




#SBATCH --job-name sam_train1_coco


##################
# Define variables

##############
# Sync data

##############
# Prepare the environment

### This worked on Intel nodes ###
# ml \
#     mmdet/3.3.0-foss-2023a-CUDA-12.1.1 \
#     mmpose/1.3.0-foss-2023a-CUDA-12.1.1 \
#     scikit-learn/1.3.1-gfbf-2023a \
#     tqdm/4.66.1-GCCcore-12.3.0 \
#     Shapely/2.0.1-gfbf-2023a \
#     tensorboard/2.15.1-foss-2023a

  


# Oneliner for debugging
# ml mmdet/3.3.0-foss-2023a-CUDA-12.3.0 mmpose/1.3.1-foss-2023a-CUDA-12.3.0 scikit-learn/1.3.1-gfbf-2023a tqdm/4.66.1-GCCcore-12.3.0 Shapely/2.0.1-gfbf-2023a tensorboard/2.15.1-foss-2023a
# python -c "import numpy as np; print(np.__file__); print(np.ndarray)"

# pip3 install sparsemax


# cd /mnt/personal/purkrmir/mmpose-ViTPose/


# Run this only the first time job is run on RCI
# pip3 install -r requirements.txt
# pip3 install -e .
# cd ../mmpretrain/
# pip3 install -r requirements.txt
# pip3 install -e .
# cd ../mmpose-ViTPose/


##############
# Show the environment


echo "--- Environment set up ---"

ml purge

module load PyTorch/2.6.0-foss-2023b-CUDA-12.4.0 \ OpenCV/4.9.0-foss-2023b-contrib \ torchvision/0.21.0-foss-2023b-CUDA-12.4.0 \ pycocotools/2.0.7-foss-2023b \ Hydra/1.3.2-GCCcore-13.2.0 \ typing-extensions/4.11.0-GCCcore-13.2.0 \ matplotlib/3.9.2-gfbf-2024a \ torchrl/0.9.2-foss-2023b-CUDA-12.4.0 \ huggingface-hub/0.29.3-GCCcore-13.2.0 \ timm/1.0.14-foss-2023b-CUDA-12.4.0
pip3 install -e ".[dev]"
echo "--- Environment set up ---"
ml list
##############
# Run the trainingcan i 

# ulimit -n
# ulimit -n 4096
# ulimit -n
# export NCCL_TIMEOUT=1200

python training/train.py -c configs/sam2.1_training/train_coco_config_maskdec.yaml --use-cluster 0 --num-gpus 4
