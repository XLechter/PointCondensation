# PointCondensation
Codes for paper 'Informative Point cloud Dataset Extraction for Classification via Gradient-based Points Moving'. ACM MM 24.

# Abstract
Point cloud plays a significant role in recent learning-based vision tasks, which contain additional information about the physical space compared to 2D images. However, such a 3D data format also results in more expensive training costs to train a sophisticated network with large 3D datasets. 
Previous methods for point cloud compression focus on compacting the representation of each point cloud for better storage and transmission but ignore the improvements in training efficiency. In this paper, we introduce a new open problem in the point cloud field, named \textit{point cloud condensation}: Can we condense a large point cloud dataset into a much smaller synthetic dataset while preserving the important information of the original large dataset? In other words, we explore the possibility of training a network on a smaller dataset of informative point clouds extracted from the original large dataset but maintaining similar network classification performance. Training on this small synthetic dataset could largely improve the training efficiency. To achieve this goal, we propose a two-stage approach to generate the synthetic dataset. We first introduce a nearest-feature-mean based strategy to initialize the synthetic dataset, and then formulate our goal as a parameter-matching problem, which we solve by introducing a gradient-matching strategy to iteratively refine the synthetic dataset. We conduct extensive experiments on various synthetic and real-scanned 3D object classification benchmarks, showing that our synthetic dataset could achieve almost the same performance with only 5\% point clouds of ScanObjectNN dataset compared to training with the full dataset. 
![image](https://github.com/user-attachments/assets/8379075c-60fa-4b88-8fc0-d2e536bb80cf)


I have uploaded a coarse version, and I will clean up the codes upon my availability.

# Requirements
You can run it on a general CUDA environment without installing further libraries if you only use PointNet as the backbone. For advanced backbones like PointNet++/KPConv, etc, you need to prepare the settings with their instructions.

# Training
Run
```
# For ModelNet10
python main_DM_pointnet.py  --dataset ModelNet  --model PointNet  --ratio 0.0001 --num_exp 1  --num_eval 5 --data_path . --dsa_strategy None --init real
python main_DM_pointnet.py  --dataset ModelNet  --model PointNet  --ratio 0.01 --num_exp 1  --num_eval 5 --data_path . --dsa_strategy None --init real

# For ModelNet40
python main_DM_pointnet.py  --dataset ModelNet40  --model PointNet  --ratio 0.0001 --num_exp 1  --num_eval 5 --data_path . --dsa_strategy None --init real
python main_DM_pointnet.py  --dataset ModelNet40  --model PointNet  --ratio 0.01 --num_exp 1  --num_eval 5 --data_path . --dsa_strategy None --init real

# For ScanObjectNN
python main_DM_pointnet.py  --dataset ScanObjectNN  --model PointNet  --ratio 0.0001 --num_exp 1  --num_eval 5 --data_path . --dsa_strategy None --init real
python main_DM_pointnet.py  --dataset ScanObjectNN  --model PointNet  --ratio 0.01 --num_exp 1  --num_eval 5 --data_path . --dsa_strategy None --init real

# --ratio: 0.0001(1 per class), 0.01(1% per class), 0.05(5% per class) --dataset: ModelNet(refers to ModelNet10), ModelNet40, ScanObjectNN
```


# Dataset Preparation

For ModelNet10, I have uploaded an h5 file in the repo, you can use it without any modification.

For ModelNet40 and ScanObjectNN, you can download them in their official repos, I will write an instruction soon.

# Visualization

Uncomment Line 215-221 in main_DM_pointnet.py

# Acknowledgement
This repo is largely based on [DC&DM](https://github.com/VICO-UoE/DatasetCondensation).

