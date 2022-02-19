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
            nn.Linear(2*hidden_size, 64),
            nn.ReLU(True),
            nn.Linear(64,64),
            nn.ReLU(True),
            nn.Linear(64,1),
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

def build_conditional_nf(n_sequence, hidden_size, state_dim, init_var=0.01, prior_mean=0.0, prior_std=1.0):
    flows = [RealNVP_cond(dim=state_dim, obser_dim=hidden_size) for _ in range(n_sequence)]

    for f in flows:
        f.zero_initialization(var=init_var)

    prior_init = MultivariateNormal(torch.zeros(state_dim).to(device) + prior_mean,
                                    torch.eye(state_dim).to(device) * prior_std**2)

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

class measurement_model_cosine_distance(nn.Module):
    def __init__(self, particle_encoder):
        super().__init__()
        self.particle_encoder = particle_encoder

    def forward(self, encodings, update_particles):
        particle_encoder = self.particle_encoder.float()
        encodings_state = particle_encoder(update_particles.float()) # shape: (batch, particle_num, hidden_size)

        encodings_obs = encodings[:, None, :].repeat(1, update_particles.shape[1],1)  # shape: (batch_size, particle_num, hidden_size)

        likelihood = 1 / (1e-7 + et_distance(encodings_obs, encodings_state))

        return likelihood.log()

class measurement_model_NN(nn.Module):
    def __init__(self, particle_encoder, likelihood_estimator):
        super().__init__()
        self.particle_encoder = particle_encoder
        self.likelihood_estimator = likelihood_estimator

    def forward(self, encodings, update_particles):
        particle_encoder = self.particle_encoder.float()
        encodings_state = particle_encoder(update_particles.float()) # shape: (batch, particle_num, hidden_size)

        encodings_obs = encodings[:, None, :].repeat(1, update_particles.shape[1],1)  # shape: (batch_size, particle_num, hidden_size)

        likelihood = self.likelihood_estimator(torch.cat([encodings_obs, encodings_state], dim=-1))

        return likelihood[..., 0].log()

class measurement_model_Gaussian(nn.Module):
    def __init__(self, particle_encoder, gaussian_distribution):
        super().__init__()
        self.particle_encoder = particle_encoder
        #gaussian_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(noise_feature.shape[-1]), torch.eye(noise_feature.shape[-1])).to(device)
        self.gaussian_distribution = gaussian_distribution
    def forward(self, encodings, update_particles):
        particle_encoder = self.particle_encoder.float()
        encodings_state = particle_encoder(update_particles.float()) # shape: (batch, particle_num, hidden_size)

        encodings_obs = encodings[:, None, :].repeat(1, update_particles.shape[1],1)  # shape: (batch_size, particle_num, hidden_size)

        noise_feature = encodings_obs - encodings_state

        likelihood = self.gaussian_distribution.log_prob(noise_feature)
        likelihood = likelihood - likelihood.max(dim=-1, keepdims=True)[0]

        return likelihood

class measurement_model_cnf(nn.Module):
    def __init__(self, particle_encoder, CNF):
        super().__init__()
        self.particle_encoder = particle_encoder
        self.CNF = CNF

    def forward(self, encodings, update_particles):
        hidden_dim = encodings.shape[-1]
        n_batch, n_particles = update_particles.shape[:2]

        particle_encoder = self.particle_encoder.float()
        encodings_state = particle_encoder(update_particles.float()) # shape: (batch, particle_num, hidden_size)
        encodings_state = encodings_state.reshape([-1, hidden_dim])

        encodings_obs = encodings[:, None, :].repeat(1, update_particles.shape[1],1)
        encodings_obs = encodings_obs.reshape([-1, hidden_dim])

        z, log_prob_z, log_det = self.CNF.forward(encodings_obs, encodings_state)
        #print(z[0].abs().mean(dim=0), z[20].abs().mean(dim=0), z[10].abs().mean(dim=0), z[30].abs().mean(dim=0), log_prob_z.mean())
        likelihood = (log_prob_z + log_det).reshape([n_batch, n_particles])
        likelihood = likelihood - likelihood.max(dim=-1, keepdims=True)[0]

        return likelihood

def nf_dynamic_model(dynamical_nf, dynamic_particles, jac_shape, NF=False, forward=False, mean=None, std=None):
    if NF:
        n_batch, n_particles, dimension = dynamic_particles.shape
        if not forward:
            dyn_particles_mean, dyn_particles_std = dynamic_particles.mean(dim=1, keepdim=True).detach().clone().repeat([1, n_particles, 1]), \
                                                    dynamic_particles.std(dim=1, keepdim=True).detach().clone().repeat([1, n_particles, 1])
        else:
            dyn_particles_mean, dyn_particles_std = mean.detach().clone().repeat([1, n_particles, 1]),\
                                                    std.detach().clone().repeat([1, n_particles, 1])
        dyn_particles_mean_flatten, dyn_particles_std_flatten = dyn_particles_mean.reshape(-1, dimension), dyn_particles_std.reshape(-1,dimension)
        context = torch.cat([dyn_particles_mean_flatten, dyn_particles_std_flatten], dim=-1)
        #dynamic_particles = (dynamic_particles - dyn_particles_mean) / dyn_particles_std

        particles_pred_flatten = dynamic_particles.reshape(-1, dimension)

        if forward:
            particles_update_nf, _, log_det = dynamical_nf.forward(particles_pred_flatten, context)
        else:
            particles_update_nf, log_det = dynamical_nf.inverse(particles_pred_flatten, context)
        jac_dynamic = -log_det
        jac_dynamic = jac_dynamic.reshape(dynamic_particles.shape[:2])

        nf_dynamic_particles = particles_update_nf.reshape(dynamic_particles.shape)
        #nf_dynamic_particles = nf_dynamic_particles * dyn_particles_std + dyn_particles_mean
    else:
        nf_dynamic_particles=dynamic_particles
        jac_dynamic = torch.zeros(jac_shape).to(device)
    return nf_dynamic_particles, jac_dynamic

def normalising_flow_propose(cond_model, particles_pred, obs, flow=RealNVP_cond, n_sequence=2, hidden_dimension=8, obser_dim=None):

    B, N, dimension = particles_pred.shape

    pred_particles_mean, pred_particles_std = particles_pred.mean(dim=1, keepdim=True).detach().clone().repeat([1, N, 1]), \
                                            particles_pred.std(dim=1, keepdim=True).detach().clone().repeat([1, N, 1])
    dyn_particles_mean_flatten, dyn_particles_std_flatten = pred_particles_mean.reshape(-1, dimension), pred_particles_std.reshape(-1, dimension)
    context = torch.cat([dyn_particles_mean_flatten, dyn_particles_std_flatten], dim=-1)
    #particles_pred = (particles_pred - pred_particles_mean) / pred_particles_std

    particles_pred_flatten=particles_pred.reshape(-1,dimension)
    obs_reshape = obs[:, None, :].repeat([1,N,1]).reshape(B*N,-1)
    obs_reshape = torch.cat([obs_reshape, context], dim=-1)

    particles_update_nf, log_det=cond_model.inverse(particles_pred_flatten, obs_reshape)

    jac=-log_det
    jac=jac.reshape(particles_pred.shape[:2])

    particles_update_nf=particles_update_nf.reshape(particles_pred.shape)
    #particles_update_nf = particles_update_nf * pred_particles_std + pred_particles_mean

    return particles_update_nf, jac

def proposal_likelihood(cond_model, dynamical_nf, measurement_model, particles_dynamical, particles_physical,
                        encodings, noise, jac_dynamic, NF, NF_cond, prototype_density):
    encodings_clone = encodings.detach().clone()
    encodings_clone.requires_grad = False

    if NF_cond:
        propose_particle, jac_prop = normalising_flow_propose(cond_model, particles_dynamical, encodings_clone)
        if NF:
            particle_prop_dyn_inv, jac_prop_dyn_inv = nf_dynamic_model(dynamical_nf, propose_particle,jac_dynamic.shape, NF=NF, forward=True,
                                                                       mean=particles_physical.mean(dim=1, keepdim=True),
                                                                       std=particles_physical.std(dim=1, keepdim=True))
            prior_log = prototype_density(particle_prop_dyn_inv - (particles_physical - noise)) - jac_prop_dyn_inv #####
        else:
            prior_log = prototype_density(propose_particle - (particles_physical - noise))
        propose_log = prototype_density(noise) + jac_dynamic + jac_prop
    else:
        propose_particle = particles_dynamical
        prior_log = prototype_density(noise) + jac_dynamic
        propose_log = prototype_density(noise) + jac_dynamic

    lki_log = measurement_model(encodings, propose_particle)
    return propose_particle, lki_log, prior_log, propose_log