# Normalizing-Flows-DPFs

This repository provides the code to reproduce the disk tracking experimental results in the paper **[Differentiable Particle Filters through Conditional Normalizing Flow](https://arxiv.org/abs/2107.00488)** and the paper **Conditional Measurement Density Estimation in Sequential Monte Carlo Methods via Normalizing Flow**.

## Problem statement in the experiment

This experiment evaluates the performance of ***[Conditional Normalizing Flow DPF (CNF-DPF)](https://arxiv.org/abs/2107.00488)*** and ***DPF Conditional normalizing flow Measurement model (DPF-CM)*** in a disk tracking experiment where the task is to track a moving red disk moving along the other distractor disks with different colours. While tracking the target, the observation images are provided at each time step, from which we expect our filter to infer the position of the red disk. The red disk can be occluded by the distractors and may occasionlly run out of the boundary of the images as collisions are not considered in our setting. More detailed description can be found in the paper.
<details>
<summary>An example of the observation image</summary>
    
<img src="https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/Figure/Disk.PNG" width="300" />
   
</details>

## Prerequisites

### Python Packages 

To install the required python packages, run the following command:

```
pip install -r requirements.txt
```

### Create datasets

Run the file [./data/disk/create_toy_dataset.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/data/disk/create_toy_dataset.py) to create the disk tracking dataset for training, validation and testing sets. The generated dataset will be stored in the folder ```./data/disk/``` as default. Some optional parameters are listed as follows:
- ```--num-distractors``` number of distractors in the observation image. 
- ```--pos_noise``` standard deviation of target positions when generating trajectories.
- ```--num_examples``` number of trajectory samples being generated. 
- ```--sequence_length``` length of generated trajectories.
- ```--out_dir``` specify the directory to store the generated datasets.

## Project Structure

### Folders

- ```./data/``` contains python scripts for creating training, validation, and testing sets used in the experiment. Generated datasets are also stored in this folder.
- ```./model/``` provides functions used to build components of evaluated models, including their dynamic model, measurement model, and proposal distributions.
- ```./nf/``` different types of flow models ([Planar Flow, Radial Flow](https://arxiv.org/abs/1505.05770), [RealNVP](https://arxiv.org/abs/1605.08803), [MAF](https://arxiv.org/abs/1705.07057), and [CGLOW](https://ojs.aaai.org/index.php/AAAI/article/view/5940) etc.) can be found in this folder.
- ```./resamplers/``` provides implementations of differentiable resampling schemes including [soft resampling](https://arxiv.org/abs/1805.08975) and [resampling via entropy-regularized optimal transport](http://proceedings.mlr.press/v139/corenflos21a/corenflos21a.pdf).
- ```./logs/``` experiment results are saved in this folder.

### Python scripts

- [experiment_DiskTracking.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/experiment_DiskTracking.py): run this file to train, validate, and test the model.
- [DPFs.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/DPFs.py): implementation of different differentiable particle filtering algorithms evaluated in the papers, contains functions for training, validating, and testing these methods.
- [dataset.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/dataset.py): create a pytorch Dataset object for the disk tracking dataset.
- [losses.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/losses.py): different optimisation objectives for training the model.
- [utils.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/utils.py): useful functions to save space in the main file.
- [arguments.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/arguments.py): configurations of the experiment.


## Arguments in the experiment
<details>
<br/>
<summary>Basic arguments in the experiment</summary>
    
- ```--trainType``` whether to train the model in the supervised setting or the semi-supervised setting, available options: **DPF|SDPF**.
- ```--lr``` learning rate when optimising the trained model.
- ```--batchsize``` batch size of training dataloader.
- ```--num_epochs``` number of training epochs.
- ```--resampler_type``` type of particle resampler used in the experiment, available options: **soft|ot**.
- ```--pos-noise``` position noise in the dynamic model.
- ```--vel-noise``` action noise in the dynamic model.
- ```--num-particles``` number of particles used in the experiment.
- ```--testing``` test a model saved in a specified path.
- ```--model-path``` path of the tested model.
    
</details>

<details>
<summary>Arguments for the CNF-DPF setup</summary>
<br/>

The CNF-DPF is proposed in the paper **[Differentiable Particle Filters through Conditional Normalizing Flow](https://arxiv.org/abs/2107.00488)**, it adopts (conditional) normalizing flows to construct flexible dynamic models (proposal distributions). Particularly, (conditional) Real-NVP is applied in the CNF-DPF.
    
To evaluate the performance of the CNF-DPF, run the following command:

```
python experiment_DiskTracking.py --NF-dyn --NF-cond 
```

Related arguments:
    
- ```--NF-dyn``` set as **True** to enhance the dynamic model with normalizing flows, default as **False**.
- ```--NF-cond``` set as **True** to propose particles using conditional normalizing flows, default as **False**.

</details>

<details>
<summary>Arguments for the DPF-CM setup</summary>
<br/>

The DPF-CM is proposed in the paper **Conditional Measurement Density Estimation in Sequential Monte Carlo Methods via Normalizing Flow**, where conditional normalizing flows are employed to estimate the likelihood of observations given states. 
    
To reproduce the epxeriment results of the DPF-CM reported in the paper **Conditional Measurement Density Estimation in Sequential Monte Carlo Methods via Normalizing Flow**, run the following command:

```
python experiment_DiskTracking.py --measurement CRNVP
```

Related arguments:
    
- ```--measurement``` select the measurement model for the evaluated methods, available options: **|CRNVP|cos|NN|CGLOW|gaussian|**

For measurement models built with conditional normalizing flows, both conditional Real-NVP and conditional-GLOW are available options in this project, but only the performance of conditional Real-NVP is reported in the paper since conditional GLOW was found to produce slightly higher prediction error than conditional Real-NVP, we are now analysing intermediate results to find out the reason for this. The **cos** option refers to the measurement model proposed in the paper **[End-to-End Semi-supervised Learning for Differentiable Particle Filters
](https://ieeexplore.ieee.org/abstract/document/9561889)**
    
</details>

## References 
### Code
The implemention of the included flow models are from this nice repository: [normalizing-flows](https://github.com/tonyduan/normalizing-flows) by [tonyduan](https://github.com/tonyduan).

## Citation
If you find this code useful for your research, please cite our paper:
```
@INPROCEEDINGS{chen2021,
    author={Chen, Xiongjie and Wen, Hao and Li, Yunpeng},
    booktitle={Proc. Intl. Conf. Information Fusion (FUSION)},
    title={Differentiable Particle Filters through Conditional Normalizing Flow},
    year={2021},
    address={Sun City, South Africa},
    month={Nov.}
}
```
