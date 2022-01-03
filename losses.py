import torch
from torch import nn
import numpy as np

def autoencoder_loss(image, train, encoder, decoder):
    mse = nn.MSELoss()

    batch, seq, c, h, w = image.shape
    image_input = torch.reshape(image, (batch * seq, c, h, w))

    feature = encoder(image_input)
    recontr_img = decoder(feature)

    loss = mse(recontr_img, image_input)

    return loss

def supervised_loss(particle_list, particle_weight_list, true_state, mask, train, labeledRatio=1.0):

    prediction = torch.sum(particle_list * particle_weight_list[:, :, :, None],
                          dim=2)  # the dataset has extra initial state
    # loss = torch.sqrt(torch.mean((prediction - true_state) ** 2))  # Rooted mean square error
    if train:
        if labeledRatio > 0:
            loss = torch.sqrt( torch.mean(mask[:,:,None]*(prediction - true_state[:,:,:2]) ** 2)/ labeledRatio ) # Rooted mean square error
            return loss,prediction
        elif labeledRatio == 0:
            return 0
    else:
        loss = torch.sqrt(torch.mean((prediction - true_state[:,:,:2]) ** 2))
        return loss,prediction

def pseudolikelihood_loss_nf(particle_weight_list, noise_list, likelihood_list, index_list, jac_list, prior_list, block_len=10):

    return -1. * torch.mean(compute_block_density_nf(particle_weight_list, noise_list, likelihood_list, index_list, jac_list, prior_list, block_len))

def compute_block_density_nf(particle_weight_list, noise_list, likelihood_list, index_list, jac_list, prior_list, block_len=10):
    batch_size, seq_len, num_resampled = particle_weight_list.shape

    # log_mu_s shape: (batch_size, num_particle)
    # block index
    b =0
    # pseudo_likelihood
    Q =0
    logyita = 0
    log_c = - 0.5 * torch.log(torch.tensor(2 * np.pi))
    for k in range(seq_len):
        if (k+1)% block_len==0:
            for j in range(k, k-block_len, -1):
                if j == k:
                    lik_log = likelihood_list[:,j,:]
                    index_a = index_list[:,j,:]
                    jac_log = jac_list[:,j,:]
                    prior_ = prior_list[:, j, :]
                else:
                    lik_log = likelihood_list[:,j,:].reshape((batch_size * num_resampled,))[index_a]
                    jac_log = jac_list[:,j,:].reshape((batch_size * num_resampled,))[index_a]
                    prior_ = prior_list[:,j,:].reshape((batch_size * num_resampled,))[index_a]

                    index_pre = index_list[:, j, :]
                    index_a = index_pre.reshape((batch_size * num_resampled,))[index_a]

                log_prior = prior_

                logyita = logyita + log_prior + lik_log
            Q = Q + torch.sum(particle_weight_list[:, k, :] * logyita, dim=-1)
            b = b+1
    # Q shape: (batch_size,)
    return Q/b


def compute_block_density(particle_weight_list, noise_list, likelihood_list, index_list, block_len=10, std_pos=1.0, std_vel=1.0):
    batch_size, seq_len, num_resampled = particle_weight_list.shape

    # log_mu_s shape: (batch_size, num_particle)
    # block index
    b =0
    # pseudo_likelihood
    Q =0
    logyita = 0
    log_c = - 0.5 * torch.log(torch.tensor(2 * np.pi))
    for k in range(seq_len):
        if (k+1)% block_len==0:
            for j in range(k, k-block_len, -1):
                if j == k:
                    lik = likelihood_list[:,j,:]
                    index_a = index_list[:,j, :]
                    noise_pos = noise_list[:, j, :, :2]
                    noise_vel = noise_list[:, j, :, 2:]
                else:
                    lik = likelihood_list[:,j,:].reshape((batch_size * num_resampled,))[index_a]
                    noise_pos = noise_list[:, j, :, :2].reshape((batch_size * num_resampled,-1))[index_a,:]
                    noise_vel = noise_list[:, j, :, 2:].reshape((batch_size * num_resampled, -1))[index_a, :]
                    index_pre = index_list[:, j, :]
                    index_a = index_pre.reshape((batch_size * num_resampled,))[index_a]

                log_prior = (2 * log_c -  2*torch.log(torch.tensor(std_pos)) - torch.sum(
                    noise_pos ** 2 / (2 * torch.tensor(std_pos) ** 2), dim=-1)) +\
                            (2 * log_c -  2*torch.log(torch.tensor(std_vel)) - torch.sum(
                            noise_vel ** 2 / (2 * torch.tensor(std_vel) ** 2), dim=-1))

                logyita = logyita + log_prior + lik
            Q = Q + torch.sum(particle_weight_list[:, k, :] * logyita, dim=-1)
            b = b+1
    # Q shape: (batch_size,)
    return Q/b



def pseudolikelihood_loss(particle_weight_list, noise_list, likelihood_list, index_list, block_len=10, std_pos=1.0, std_vel=1.0):

    return -1. * torch.mean(compute_block_density(particle_weight_list, noise_list, likelihood_list, index_list, block_len, std_pos, std_vel))