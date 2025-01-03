import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def et_distance(encoding_input,e_t):

    # tf.reduce_mean((encoding_input-e_t)**2,axis=-1)
    encoding_input=F.normalize(encoding_input,p=2,dim=-1,eps=1e-12)
    e_t=F.normalize(e_t,p=2,dim=-1, eps=1e-12)
    cosd = 1.0 - torch.sum(encoding_input*e_t,dim=-1)

    return cosd

class compute_normal_density(nn.Module):
    def __init__(self, pos_noise=1.0, vel_noise=1.0):
        super().__init__()
        self.pos_noise=pos_noise
        self.vel_noise=vel_noise
    def forward(self, noise, std_pos = None, std_vel = None):
        log_c = - 0.5 * torch.log(torch.tensor(2 * np.pi)) 
        if std_pos is None:
            std_pos = self.pos_noise
        if std_vel is None:
            std_vel = self.vel_noise

        noise_pos = noise[:, :, :2]
        noise_vel = noise[:, :, 2:]

        log_prior = noise.shape[-1] * log_c - 2 * torch.log(torch.tensor(std_pos)) - torch.sum(
            noise_pos ** 2 / (2 * torch.tensor(std_pos) ** 2), dim=-1) + \
                    - (noise.shape[-1]-2) * torch.log(torch.tensor(std_vel)) - torch.sum(
            noise_vel ** 2 / (2 * torch.tensor(std_vel) ** 2), dim=-1)

        return log_prior

def normalize_log_probs(probs):
    probs_max=probs.max(dim=1, keepdims=True)[0]
    probs_minus_max = probs-probs_max
    probs_normalized = probs_minus_max.exp()
    probs_normalized = probs_normalized / torch.sum(probs_normalized, dim=1, keepdim=True)
    return probs_normalized

def particle_initialization(start_state, width, num_particles, state_dim=2,init_with_true_state=False):
    batch_size = start_state.shape[0]
    if init_with_true_state:
        initial_noise = torch.randn(batch_size, num_particles, state_dim).to(device)
        initial_particles = start_state[:, None, :].repeat(1, num_particles, 1) + initial_noise
    else:
        bound_max = width / 2.0
        bound_min = -width / 2.0
        pos = torch.tensor((bound_max - bound_min)).to(device) * torch.rand(batch_size, num_particles,
                                                                            2).to(device) + torch.tensor(
            bound_min).to(device)
        initial_particles = pos
        vel = torch.randn(batch_size, num_particles, 2).to(device)

    init_weights_log = torch.log(torch.ones([batch_size, num_particles]).to(device) / num_particles)

    return initial_particles, init_weights_log

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def checkpoint_state(model,epoch):
    state_dict={
        "model": model.state_dict(),
        'model_optim': model.optim.state_dict(),
        'model_optim_scheduler': model.optim_scheduler.state_dict(),
        "epoch": epoch
    }
    return state_dict

def load_model(model, ckpt_e2e):
    model.load_state_dict(ckpt_e2e['model'])
    model.optim.load_state_dict(ckpt_e2e['model_optim'])
    model.optim_scheduler.load_state_dict(ckpt_e2e['model_optim_scheduler'])