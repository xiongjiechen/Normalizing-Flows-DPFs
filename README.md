# Normalizing-Flows-DPFs

This repository provides the code to reproduce the experimental results in the paper **[Normalizing Flow-based Differentiable Particle Filters
](https://arxiv.org/abs/2403.01499)**.

## Problem statement in this experiment

The experiments evaluate the performance of the Normalizing Flow-based Differentiable Particle Filters (NF-DPFs) in different setups. More detailed descriptions of the experiment setups can be found in the paper **[Normalizing Flow-based Differentiable Particle Filters
](https://arxiv.org/abs/2403.01499)**.

## Prerequisites

### Install python Packages 

To install the required python packages, run the following command:

```
pip install -r requirements.txt
```

### Prepare datasets

To create datasets for the training and testing, please run the file [create_toy_dataset.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/data/disk/create_toy_dataset.py): 

```
python create_toy_dataset.py
``` 

Some optional parameters are listed as follows:
- ```--pos-noise``` standard deviation of target positions when generating trajectories.
- ```--vel-noise``` action noise of the target object.
- ```--num_examples``` number of trajectory samples being generated. 
- ```--sequence_length``` length of generated trajectories.
- ```--out_dir``` specifies the directory to store the generated dataset.

## Project Structure

### Folders

- ```./data/``` contains python scripts for creating training, validation, and testing sets used in the experiment. Generated datasets are also stored in this folder.
- ```./model/``` provides functions used to build components of evaluated models, including their dynamic model, measurement model, and proposal distributions.
- ```./nf/``` different types of flow models ([Planar Flow, Radial Flow](https://arxiv.org/abs/1505.05770), [RealNVP](https://arxiv.org/abs/1605.08803), [MAF](https://arxiv.org/abs/1705.07057), and [CGLOW](https://ojs.aaai.org/index.php/AAAI/article/view/5940) etc.) can be found in this folder.
- ```./resamplers/``` provides implementations of differentiable resampling schemes including [soft resampling](https://arxiv.org/abs/1805.08975) and [resampling via entropy-regularized optimal transport](http://proceedings.mlr.press/v139/corenflos21a/corenflos21a.pdf).
- ```./logs/``` experiment results are saved in this folder.

### Python scripts

- [main.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/main.py): run this file to train, validate, and test the model.
- [DPFs.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/DPFs.py): implementation of different differentiable particle filtering algorithms evaluated in the papers, contains functions for training, validating, and testing these methods.
- [dataset.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/dataset.py): define a pytorch Dataset object for the created dataset.
- [losses.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/losses.py): different optimisation objectives for training the model.
- [utils.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/utils.py): useful functions to save space in the main file.
- [arguments.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/arguments.py): configurations of the experiment.


## Arguments in the experiment
#### Basic arguments in the experiment
    
- ```--trainType``` whether to train the model in the supervised setting or the semi-supervised setting.
- ```--lr``` learning rate when optimising the trained model.
- ```--batchsize``` batch size of training dataloader.
- ```--num_epochs``` number of training epochs.
- ```--resampler_type``` type of particle resampler used in the experiment.
- ```--num-particles``` number of particles used in the experiment.
- ```--testing``` test a model saved in a specified path.
- ```--model-path``` path of the tested model.
    

<details>
<summary>Arguments for the CNF-DPF setup</summary>
<br/>

The CNF-DPF is proposed in the paper **[Differentiable Particle Filters through Conditional Normalizing Flow](https://arxiv.org/abs/2107.00488)**, it adopts (conditional) normalizing flows to construct flexible dynamic models (proposal distributions). Particularly, (conditional) Real-NVP is applied in the CNF-DPF.
    
To evaluate the performance of the CNF-DPF, run the following command:

```
python main.py --NF-dyn --NF-cond 
```

Related arguments:
    
- ```--NF-dyn``` set as **True** to enhance the dynamic model with normalizing flows, default as **False**.
- ```--NF-cond``` set as **True** to propose particles using conditional normalizing flows, default as **False**.

</details>

<details>
<summary>Arguments for the DPF-CM setup</summary>
<br/>

The DPF-CM is proposed in the paper **Conditional Measurement Density Estimation in Sequential Monte Carlo Methods via Normalizing Flow**, where conditional normalizing flows are employed to estimate the likelihood of observations given states. 
    
To reproduce the epxeriment results of the DPF-CM reported in the paper, run the following command:

```
python main.py --measurement CRNVP
```
    
#### Available measurement models
    
- ```--measurement``` selects the measurement model for the evaluated methods, available options: **| cos | gaussian | NN | CGLOW | CRNVP |**
    <br/>
    
    -  For measurement models built with conditional normalizing flows, both conditional Real-NVP **(CRNVP)** and conditional-GLOW **(CGLOW)** are available options in this project, but only the performance of conditional Real-NVP is reported in the paper since conditional GLOW was found to produce slightly higher prediction error (RMSE) than conditional Real-NVP, we are now analysing intermediate results to find out the reason for this.     
    - The option **| cos |** refers to the measurement model proposed in the paper **[End-to-End Semi-supervised Learning for Differentiable Particle Filters](https://ieeexplore.ieee.org/abstract/document/9561889)**, where the likelihood of observation given states is estimated by the cosine similarity between the observation feature and the state feature. 
    - The option **| gaussian |** is the measurement model used in the robot localization experiment in the paper **[Differentiable particle filtering via entropy-regularized optimal transport](http://proceedings.mlr.press/v139/corenflos21a.html)**. This measurement model estimates the likelihood by computing the Gaussian density of observation feature conditioned on the state feature.    
    - The option **| NN |** denotes the measurement model proposed in the paper **[Particle Filter Networks with Application to Visual Localization
](https://arxiv.org/abs/1805.08975)**, it considers the observation likelihoods as the outputs of a neural network , whose input is the concatenation of feature maps of observations and states.

Code for the above measurement models can be found in the python script ```./model/models.py```
    
</details>

<span style="color: blue">If you used other values other than the default one for the argument ```--pos-noise``` when running the script [create_toy_dataset.py](https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/data/disk/create_toy_dataset.py), it is needed to accordinlgy set the argument ```--true-pos-noise``` as the same value when running the main file, otherwise the Dataloader won't be able to find the dataset files.</span>

## References 
### Code
The implemention of included normalizing flow models are from the repository: [normalizing-flows](https://github.com/tonyduan/normalizing-flows) by [tonyduan](https://github.com/tonyduan), the implementation of the conditional GLOW model is based on the repository [Conditional-GLOW](https://github.com/yolu1055/conditional-glow) by [You Lu](https://github.com/yolu1055).

## Citation
If you find this code useful for your research, please cite our paper:
```
@article{chen2024,
    author={Chen, Xiongjie and and Li, Yunpeng},
    booktitle={IEEE Trans. Signal Process. (TSP)},
    title={Normalizing Flow-based Differentiable Particle Filters},
    year={2024}
}
```
```
@inproceedings{chen2021,
    author={Chen, Xiongjie and Wen, Hao and Li, Yunpeng},
    booktitle={Proc. Intl. Conf. Information Fusion (FUSION)},
    title={Differentiable Particle Filters through Conditional Normalizing Flow},
    year={2021},
    address={Sun City, South Africa},
    month={Nov.}
}
```
```
@inproceedings{chen2022,
    author={Chen, Xiongjie and and Li, Yunpeng},
    booktitle={Proc. Euro. Sig. Process. Conf. (EUSIPCO)},
    title={Conditional Measurement Density Estimation in Sequential Monte Carlo Methods via Normalizing Flow},
    year={2022},
    address={Belgrade, Serbia},
    month={Aug.}
}
```
