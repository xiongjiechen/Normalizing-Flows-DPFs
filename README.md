# Normalizing-Flows-DPFs

This repository provides the code to reproduce the disk tracking experimental results in the paper **[Differentiable Particle Filters through Conditional Normalizing Flow](https://arxiv.org/abs/2107.00488)** by **[Xiongjie Chen](https://scholar.google.com/citations?user=Tb9fTOsAAAAJ&hl=en&inst=15262737669262836719), [Hao Wen](https://www.surrey.ac.uk/people/hao-wen)** and **[Yunpeng Li](https://www.surrey.ac.uk/people/yunpeng-li)**.

## Prerequisite

### Python Packages 

To install the required python packages, run the following command:

```
pip install -r requirements.txt
```

## Experiment Details & Project Structure

This experiment evaluates the performance of **Conditional Normalizing Flow DPFs (CNF-DPFs)*** in a disk tracking experiment where the task is to track a moving red disk moving along the other distractor disks with different colours. While tracking the target, the observation images are provided at each time step, from which we expect our filter to infer the position of the red disk. The red disk can be occluded by the distractors and may occasionlly run out of the boundary of the images as collisions are not considered in our setting.

### Create datasets

Run the file [./data/disk/create_toy_dataset.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/data/disk/create_toy_dataset.py) to create the disk tracking dataset for training, validation and testing sets. The generated dataset will be stored in the folder ```./data/disk/``` as default. Some optional parameters are listed as follows:
- ```--num-distractors``` number of distractors in the observation image.
- ```--pos_noise``` standard deviation of target positions when generating trajectories.
- ```--num_examples``` number of trajectory samples being generated.
- ```--sequence_length``` length of generated trajectories.
- ```--out_dir``` specify the directory to store the generated datasets.
