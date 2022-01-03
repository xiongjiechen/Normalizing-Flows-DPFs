import torch
from torch import nn
from nf.models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
from util import et_distance
from nf.flows import *
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def build_encoder(hidden_size):
    encode=nn.Sequential(  # input: 3*120*120, 3*128*128
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 16*60*60, 64
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 32*30*30, 32
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64*15*15, 16
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 64*15*15, 8
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 64*15*15, 4
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            # nn.Dropout2d(p=1 - args.dropout_keep_ratio),
            nn.Linear(256 * 4 * 4, hidden_size),
            # nn.ReLU(True),
            # nn.Linear(256, self.hidden_size),
            # nn.ReLU(True) # output size: 32
        )
    return encode

def build_decoder(hidden_size):
    decode=nn.Sequential(
            nn.Linear(hidden_size, 256*4*4),
            # nn.ReLU(True),
            # nn.Linear(256, 64 * 15 * 15),
            # nn.ReLU(True),
            nn.Unflatten(-1,(256, 4, 4)), # -1 means the last dim, (64, 15, 15)

            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2, bias=False),  # (32, 30,30), 8
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),  # (32, 30,30), 16
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False),  # (32, 30,30), 32
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False),  # (16, 60,60), 64
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, kernel_size=4, padding=1, stride=2, bias=False),  # (3, 120, 120), 128
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    return decode

def build_likelihood(hidden_size, state_dim):
    likelihood=nn.Sequential(
            nn.Linear(hidden_size+state_dim, 128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    return likelihood

def build_particle_encoder(hidden_size, state_dim):
    particle_encode=nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_size),
            # nn.ReLU()
        )
    return particle_encode

def build_transition_model(state_dim):
    transition=nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),)
    return transition

def build_conditional_nf(n_sequence, hidden_size, state_dim, init_var=0.01):
    flows = [RealNVP_cond(dim=state_dim, obser_dim=hidden_size) for _ in range(n_sequence)]

    for f in flows:
        f.zero_initialization(var=init_var)

    prior_init = MultivariateNormal(torch.zeros(state_dim).to(device), torch.eye(state_dim).to(device))

    cond_model = NormalizingFlowModel_cond(prior_init, flows, device=device)

    return cond_model


def build_dyn_nf(n_sequence, hidden_size, state_dim, init_var=0.01):
    flows_dyn = [RealNVP(dim=state_dim) for _ in range(n_sequence)]

    for f in flows_dyn:
        f.zero_initialization(var=init_var)

    prior_dyn = MultivariateNormal(torch.zeros(state_dim).to(device), torch.eye(state_dim).to(device))

    nf_dyn = NormalizingFlowModel(prior_dyn, flows_dyn, device=device)

    return nf_dyn

def motion_update(particles, vel, pos_noise=20.0):
    B, N, d = particles.shape

    vel_p = vel[:, None, :].repeat((1, N, 1))

    particles_xy_update = particles[:, :, :] + vel_p

    # noise
    position_noise = torch.normal(mean=0., std=pos_noise, size=(B, N, 2)).to(
        device)

    particles_update = particles_xy_update + position_noise

    return particles_update, position_noise

class measurement_update_semi(nn.Module):
    def __init__(self, particle_encoder):
        super().__init__()
        self.particle_encoder=particle_encoder

    def forward(self, encodings, update_particles):
        particle_encoder = self.particle_encoder.float()
        e_s = particle_encoder(update_particles.float()) # shape: (batch, particle_num, hidden_size)

        # e_s =update_particles[:,:,:2]

        encodings_input = encodings[:, None, :].repeat(1, update_particles.shape[1],
                                                       1)  # shape: (batch_size, particle_num, hidden_size)

        # likelihood = 1/(1e-7+et_distance(encodings_input, e_s))
        likelihood = 1 / (1e-7 + et_distance(encodings_input, e_s))

        return likelihood

def nf_dynamic_model(dynamical_nf, dynamic_particles, jac_shape, NF=False, forward=False):
    if NF:
        dimension = dynamic_particles.shape[-1]
        particles_pred_flatten = dynamic_particles.reshape(-1, dimension)
        if forward:
            particles_update_nf, _, log_det = dynamical_nf.forward(particles_pred_flatten)
        else:
            particles_update_nf, log_det = dynamical_nf.inverse(particles_pred_flatten)
        jac_dynamic = -log_det
        jac_dynamic = jac_dynamic.reshape(dynamic_particles.shape[:2])

        nf_dynamic_particles = particles_update_nf.reshape(dynamic_particles.shape)
    else:
        nf_dynamic_particles=dynamic_particles
        jac_dynamic = torch.zeros(jac_shape).to(device)
    return nf_dynamic_particles, jac_dynamic

def normalising_flow_propose(cond_model, particles_pred, obs, flow=RealNVP_cond, n_sequence=2, hidden_dimension=8, obser_dim=None):

    B, N, dimension = particles_pred.shape

    particles_pred_flatten=particles_pred.reshape(-1,dimension)
    obs_reshape = obs[:, None, :].repeat([1,N,1]).reshape(B*N,-1)

    particles_update_nf, log_det=cond_model.inverse(particles_pred_flatten, obs_reshape)

    jac=-log_det
    jac=jac.reshape(particles_pred.shape[:2])

    particles_update_nf=particles_update_nf.reshape(particles_pred.shape)

    return particles_update_nf, jac

def proposal_likelihood(cond_model, dynamical_nf, measurement_model, particles_dynamical, particles_physical,
                        encodings, noise, jac_dynamic, NF, NF_cond, prototype_density):
    encodings_clone = encodings.detach().clone()
    encodings_clone.requires_grad = False
    particles_dynamical_clone=particles_dynamical.detach().clone()
    particles_dynamical_clone.requires_grad = False

    if NF_cond:
        propose_particle, jac_prop = normalising_flow_propose(cond_model, particles_dynamical, encodings_clone)
        particle_prop_dyn_inv, jac_prop_dyn_inv = nf_dynamic_model(dynamical_nf, propose_particle,jac_dynamic.shape, NF=NF, forward=True)
        prior_log = prototype_density(particle_prop_dyn_inv - (particles_physical - noise)) - jac_prop_dyn_inv #####
        propose_log = prototype_density(noise) + jac_dynamic + jac_prop
    else:
        propose_particle = particles_dynamical
        prior_log = prototype_density(noise) + jac_dynamic
        propose_log = prototype_density(noise) + jac_dynamic

    lki_log = measurement_model(encodings, propose_particle).log()
    return propose_particle, lki_log, prior_log, propose_log