import torch
import torch.nn as nn
import numpy as np
from util import *
from nf.flows import *
from nf.models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
import os
from torch.utils.tensorboard import SummaryWriter
from plot import *
from model.models import *
from resamplers.resamplers import resampler
from losses import *
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#6,1994715,10,311,1006,54,23,6,24,98


class DPF(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.param = args
        self.NF = args.NF_dyn
        self.NFcond = args.NF_cond
        self.measurement = args.measurement
        self.hidden_size = args.hiddensize # origin: 32
        self.state_dim = 2 #4
        self.lr = args.lr
        self.alpha=args.alpha
        self.seq_len = args.sequence_length
        self.num_particle = args.num_particles
        self.batch_size = args.batchsize

        self.labeledRatio = args.labeledRatio

        self.spring_force = 0.1 #0.1 #0.05  # 0.1 for one object; 0.05 for five objects
        self.drag_force = 0.0075 #0.0075

        self.pos_noise = args.pos_noise #0.1 #0.1
        self.vel_noise = args.vel_noise # 2.
        self.NF_lr=args.NF_lr
        self.n_sequence = 2

        self.build_model()

        self.eps=args.epsilon
        self.scaling= args.scaling
        self.threshold=args.threshold
        self.max_iter=args.max_iter
        self.resampler=resampler(self.param)

    def build_model(self):

        self.encoder = build_encoder(self.hidden_size)
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)

        self.decoder = build_decoder(self.hidden_size)
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)

        self.likelihood_est = build_likelihood(self.hidden_size, self.state_dim)
        self.likelihood_est_optim = torch.optim.Adam(self.likelihood_est.parameters(), lr=self.lr)

        self.gaussian_distribution = torch.distributions.MultivariateNormal(torch.ones(self.hidden_size).to(device),
                                                                                                100*torch.eye(self.hidden_size).to(device))

        self.particle_encoder = build_particle_encoder(self.hidden_size,self.state_dim)
        self.particle_encoder_optim = torch.optim.Adam(self.particle_encoder.parameters(), lr=self.lr)

        self.transition_model = build_transition_model(self.state_dim)
        self.transition_model_optim = torch.optim.Adam(self.transition_model.parameters(), lr=self.lr)

        self.motion_update=motion_update

        # normalising flow dynamic initialisation
        self.nf_dyn = build_conditional_nf(self.n_sequence, 2 * self.state_dim, self.state_dim, init_var=0.01)
        self.nf_dyn_optim = torch.optim.Adam(self.nf_dyn.parameters(), lr=self.NF_lr*self.lr)

        self.cond_model = build_conditional_nf(self.n_sequence, 2 * self.state_dim + self.hidden_size, self.state_dim, init_var=0.01)
        self.cond_model_optim = torch.optim.Adam(self.cond_model.parameters(), lr=self.NF_lr*self.lr)

        self.cnf_measurement = build_conditional_nf(self.n_sequence, self.hidden_size, self.hidden_size, init_var=0.01, prior_std=2.5)
        self.cnf_measurement_optim = torch.optim.Adam(self.cnf_measurement.parameters(), lr=self.lr)

        if self.measurement=='CNF':
            self.measurement_model = measurement_model_cnf(self.particle_encoder, self.cnf_measurement)
        elif self.measurement=='cos':
            self.measurement_model = measurement_model_cosine_distance(self.particle_encoder)
        elif self.measurement=='NN':
            self.measurement_model = measurement_model_NN(self.particle_encoder, self.likelihood_est)
        elif self.measurement=='gaussian':
            self.measurement_model = measurement_model_Gaussian(self.particle_encoder, self.gaussian_distribution)

        self.prototype_density=compute_normal_density(pos_noise=self.pos_noise, vel_noise= self.vel_noise)

    def forward(self, inputs, train=True):
        (start_image, start_state, image, state, q, visible) = inputs

        state = state.to(device)
        start_state = start_state.to(device)
        image = image.permute(0, 1, 4, 2, 3)  # for pytorch, the channels are above the width & height
        image = image.to(device)

        # modify the dimension of hidden state
        vel = state[:, :, 2:] + torch.normal(0.0, 4.0, (state[:, :, 2:]).shape).to(device)

        particle_list, particle_weight_list, noise_list, likelihood_list, init_weights_log, index_list, jac_list, prior_list, obs_likelihood = self.filtering_pos(image, start_state, vel)

        # mask
        if train:
            mask = self.get_mask()  # shape: (batch_size, seq_len)
        else:
            mask = 1.0

        loss_sup, predictions = supervised_loss(particle_list,particle_weight_list, state, mask, train)
        loss_ae = autoencoder_loss(image, train, self.encoder, self.decoder)

        if self.param.trainType == 'DPF':
            lamda1 = 1.0
            lamda2 = 0.01
            lamda3 = 2.0
            loss_pseud_lik = None

            total_loss = lamda1 * loss_sup + lamda3 * loss_ae #

        elif self.param.trainType == 'SDPF':
            lamda1 = 10.0
            lamda2 = 0.01
            lamda3 = 200.0
            # loss_pseud_lik = self.pseudolikelihood_loss(particle_weight_list, noise_list, likelihood_list, index_list)
            if self.NF:
                loss_pseud_lik = pseudolikelihood_loss_nf(particle_weight_list, noise_list, likelihood_list, index_list,
                                                          jac_list, prior_list, self.param.block_length)
            else:
                loss_pseud_lik = pseudolikelihood_loss(particle_weight_list, noise_list, likelihood_list, index_list, self.param.block_length,
                                                            self.param.pos_noise,self.param.vel_noise)

            total_loss = lamda1 * loss_sup + lamda2 * loss_pseud_lik + lamda3 * loss_ae
        else:
            raise ValueError('Please select the training type in DPF (supervised learning) and SDPF (semi-supervised learning)')

        return total_loss, loss_sup, loss_pseud_lik, loss_ae, predictions, particle_list, particle_weight_list, state, start_state, image, likelihood_list, noise_list, obs_likelihood

    def filtering_pos(self, obs, start_state_vs, vel_input):

        start_state = start_state_vs[:, :2]
        start_vel = start_state_vs[:, 2:]

        batch_size = start_state.shape[0]

        initial_particles, init_weights_log=particle_initialization(start_state, self.param.width, self.num_particle, self.state_dim, init_with_true_state=self.param.init_with_true_state)

        initial_particle_probs = normalize_log_probs(init_weights_log)
        obs_likelihood = 0.0

        particles = initial_particles
        particle_probs = initial_particle_probs
        vel = start_vel

        for step in range(self.seq_len):
            # index_p shape: (batch, num_p)
            index_p = (torch.arange(self.num_particle)+self.num_particle* torch.arange(batch_size)[:, None].repeat((1, self.num_particle))).type(torch.int64).to(device)
            ESS= torch.mean(1/torch.sum(particle_probs**2, dim=-1))

            if ESS<0.5*self.num_particle:
                particles_resampled, particle_probs_resampled, index_p=self.resampler(particles, particle_probs)
                particle_probs_resampled = particle_probs_resampled.log()
            else:
                particles_resampled = particles
                particle_probs_resampled = particle_probs.log()

            particles_physical, noise = self.motion_update(particles_resampled, vel, pos_noise=self.pos_noise)
            vel = vel_input[:, step, :]

            particles_dynamical, jac = nf_dynamic_model(self.nf_dyn, particles_physical,particle_probs.shape, NF=self.NF)

            encodings = self.encoder(obs[:, step].float())  # encodings shape: (batch, hidden_dim)

            propose_particle, lki_log, prior_log, propose_log = proposal_likelihood(self.cond_model,
                                                                                    self.nf_dyn,
                                                                                    self.measurement_model,
                                                                                    particles_dynamical,
                                                                                    particles_physical,
                                                                                    encodings, noise, jac,
                                                                                    self.NF, self.NFcond,
                                                                                    prototype_density= self.prototype_density)
            particle_probs_resampled = particle_probs_resampled + lki_log + prior_log - propose_log

            particles = propose_particle
            particle_probs = particle_probs_resampled
            obs_likelihood += particle_probs.mean()
            particle_probs = normalize_log_probs(particle_probs)+1e-12

            if step ==0:
                particle_list= particles[:, None, :, :]
                particle_probs_list = particle_probs[:, None, :]
                noise_list = noise[:, None, :, :]
                likelihood_list = lki_log[:, None, :]
                index_list = index_p[:, None, :]
                if self.NF:
                    jac_list = jac[:, None, :]
                    prior_list = prior_log[:, None, :]
                else:
                    jac_list = None
                    prior_list = None
            else:
                particle_list = torch.cat([particle_list, particles[:, None]], dim=1)
                particle_probs_list = torch.cat([particle_probs_list, particle_probs[:, None]], dim=1)
                noise_list = torch.cat([noise_list, noise[:, None]], dim=1)
                likelihood_list = torch.cat([likelihood_list, lki_log[:, None]], dim=1)
                index_list = torch.cat([index_list, index_p[:,None]], dim=1)
                if self.NF:
                    jac_list = torch.cat([jac_list, jac[:,None]], dim=1)
                    prior_list = torch.cat([prior_list, prior_log[:, None]], dim=1)

        return particle_list, particle_probs_list, noise_list, likelihood_list, init_weights_log, index_list, jac_list, prior_list, obs_likelihood

    def get_mask(self):

        # number of 0 and 1
        N1 = int(self.batch_size*self.seq_len*self.labeledRatio)
        N0 = self.batch_size*self.seq_len - N1
        arr = np.array([0] * N0 + [1] * N1)
        np.random.shuffle(arr)
        mask = arr.reshape(self.batch_size, self.seq_len)

        mask = torch.tensor(mask).to(device)

        return mask

    def set_train_mode(self):
        self.encoder.train()
        self.decoder.train()
        self.likelihood_est.train()
        self.particle_encoder.train()
        self.transition_model.train()

        self.cond_model.train()
        self.nf_dyn.train()
        self.cnf_measurement.train()

    def set_eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()
        self.likelihood_est.eval()
        self.particle_encoder.eval()
        self.transition_model.eval()

        self.cond_model.eval()
        self.nf_dyn.eval()
        self.cnf_measurement.eval()

    def set_zero_grad(self):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        self.likelihood_est_optim.zero_grad()
        self.particle_encoder_optim.zero_grad()
        self.transition_model_optim.zero_grad()

        self.cond_model_optim.zero_grad()
        self.nf_dyn_optim.zero_grad()
        self.cnf_measurement.zero_grad()

    def set_optim_step(self):
        self.encoder_optim.step()
        self.decoder_optim.step()
        self.likelihood_est_optim.step()
        self.particle_encoder_optim.step()
        self.transition_model_optim.step()

        self.cond_model_optim.step()
        self.nf_dyn_optim.step()
        self.cnf_measurement_optim.step()

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in self.encoder_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.decoder_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.likelihood_est_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.particle_encoder_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.transition_model_optim.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate_pretrain_ae(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = 1e-3
        for param_group in self.encoder_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.decoder_optim.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate_e2e(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr
        for param_group in self.encoder_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.decoder_optim.param_groups:
            param_group['lr'] = lr

    def pretrain_ae(self, train_loader, valid_loader, start_epoch=-1, epoch_num = 100, logger = None):

        best_eval_loss = 1e10
        best_epoch = -1

        # modify the lr for ae
        self.adjust_learning_rate_pretrain_ae()

        for epoch in range(start_epoch + 1, epoch_num):
            # train
            self.encoder.train()
            self.decoder.train()
            total_loss = []
            for batch_idx, inputs in enumerate(train_loader):
                (start_image, start_state, image, state, q, visible) = inputs

                image = image.permute(0, 1, 4, 2, 3)  # for pytorch, the channels are in front of width*height
                image = image.reshape(-1, 3, 128, 128)
                img = image.to(device)

                feature = self.encoder(img)
                recontr_img = self.decoder(feature)
                loss = F.mse_loss(recontr_img, img)

                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()

                loss.backward()

                self.encoder_optim.step()
                self.decoder_optim.step()

                print(f"Train AE: Iter: {batch_idx}, loss: {loss.detach().cpu().numpy()}")
                total_loss.append(loss.detach().cpu().numpy())

            print(f"Train AE: Epoch: {epoch}, loss: {np.mean(total_loss)}")

            # validation
            self.encoder.eval()
            self.decoder.eval()
            total_val_loss = []
            with torch.no_grad():
                for batch_idx, inputs in enumerate(valid_loader):
                    (start_image, start_state, image, state, q, visible) = inputs

                    image = image.permute(0, 1, 4, 2, 3)  # for pytorch, the channels are in front of width*height
                    (batchsize, seq_len) = image.shape[:2]
                    image = image.reshape(-1, 3, 128, 128)
                    img = image.to(device)

                    self.encoder_optim.zero_grad()
                    self.decoder_optim.zero_grad()

                    feature = self.encoder(img)
                    recontr_img = self.decoder(feature)
                    loss = F.mse_loss(recontr_img, img)

                    print(f"Evaluation AE: Iter: {batch_idx}, loss: {loss.detach().cpu().numpy()}")
                    total_val_loss.append(loss.detach().cpu().numpy())

                    plot_obs(img.reshape(batchsize, seq_len, 3, 128, 128),
                             recontr_img.reshape(batchsize, seq_len, 3, 128, 128))

                eval_loss_sup_mean = np.mean(total_val_loss)
                logger.add_scalar('PretrainAE_loss_eval/loss', eval_loss_sup_mean, epoch)
                print(f"Evaluation AE: Epoch: {epoch}, loss: {eval_loss_sup_mean}")

            # save pretain ae
            if eval_loss_sup_mean < best_eval_loss:
                best_eval_loss = eval_loss_sup_mean
                best_epoch = epoch
                print('Save best validation AE-model!')
                ckpt_ae = {
                    "encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict(),
                    'encoder_optim': self.encoder_optim.state_dict(),
                    'decoder_optim': self.decoder_optim.state_dict(),
                }
                torch.save(ckpt_ae, './model/ae_pretrain.pth')
        # load the pretrained dynamic model
        self.encoder.load_state_dict(ckpt_ae['encoder'])
        self.decoder.load_state_dict(ckpt_ae['decoder'])
        self.encoder_optim.load_state_dict(ckpt_ae['encoder_optim'])
        self.decoder_optim.load_state_dict(ckpt_ae['decoder_optim'])

    def e2e_train(self, train_loader, valid_loader, start_epoch=-1, epoch_num = 100, logger = None, run_id=None):

        params = self.param
        self.adjust_learning_rate_e2e()

        best_eval_loss = 1e10
        best_epoch = -1

        if self.param.load_pretrainModel:
            print('Load pretrained model')
            # load pretrained ae model
            ckpt_ae = torch.load('./model/ae_pretrain.pth')
            self.encoder.load_state_dict(ckpt_ae['encoder'])

            ckpt_condflow = torch.load('./model/condflow_model_bestval.pth')
            self.cond_model.load_state_dict(ckpt_condflow['model'])

        # self.unfreeze_model(self.encoder)
        unfreeze_model(self.encoder)
        # self.freeze_model(self.transition_model)

        eval_loss_epoch = []

        for epoch in range(start_epoch + 1, epoch_num):
            # train
            self.set_train_mode()
            total_sup_loss = []
            total_ae_loss = []
            for iteration, inputs in enumerate(train_loader):

                loss_all, loss_sup, loss_pseud_lik, loss_ae, predictions, particle_list, particle_weight_list, state, start_state, image, likelihood_list, noise_list, obs_likelihood = self.forward(
                    inputs, train=True)

                self.set_zero_grad()
                loss_all.backward()
                self.set_optim_step()

                ## debug
                if params.trainType == 'SDPF':
                    print(
                        f"loss_sup: {loss_sup.detach().cpu().numpy()}, loss_pseud_lik: {loss_pseud_lik.detach().cpu().numpy()}, loss_ae: {loss_ae.detach().cpu().numpy()}")

                total_sup_loss.append(loss_sup.detach().cpu().numpy())
                total_ae_loss.append(loss_ae.detach().cpu().numpy())

            train_loss_sup_mean = np.mean(total_sup_loss)
            total_ae_loss_mean = np.mean(total_ae_loss)
            logger.add_scalar('Sup_loss/loss', train_loss_sup_mean, epoch)

            print(f"End-to-end loss: epoch: {epoch}, loss: {train_loss_sup_mean}, loss_ae: {total_ae_loss_mean}, obs_likelihood: {obs_likelihood}")

            # evaluate
            self.set_eval_mode()
            total_sup_eval_loss = []

            with torch.no_grad():
                for iteration, inputs in enumerate(valid_loader):

                    self.set_zero_grad()

                    loss_all, loss_sup, loss_pseud_lik, loss_ae, predictions, particle_list, particle_weight_list, state, start_state, image, likelihood_list, noise_list,obs_likelihood = self.forward(
                        inputs, train=False)
                    total_sup_eval_loss.append(loss_sup.detach().cpu().numpy())

                eval_loss_sup_mean = np.mean(total_sup_eval_loss)
                logger.add_scalar('Sup_loss_eval/loss', eval_loss_sup_mean, epoch)
                print(f"End-to-end loss evaluation: epoch: {epoch}, loss: {eval_loss_sup_mean}, obs_likelihood: {obs_likelihood}", self.NF)

            eval_loss_epoch.append(eval_loss_sup_mean)##############
            np.save(os.path.join('logs', run_id, "data", 'eval_loss_epoch.npy'), eval_loss_epoch)

            if eval_loss_sup_mean < best_eval_loss:
                best_eval_loss = eval_loss_sup_mean
                best_epoch = epoch
                print('Save best validation model')
                np.savez(os.path.join('logs', run_id, "data", 'eval_result_best.npz'),
                         particle_list=particle_list.detach().cpu().numpy(),
                         particle_weight_list=particle_weight_list.detach().cpu().numpy(),
                         likelihood_list = likelihood_list.detach().cpu().numpy(),
                         pred=predictions.detach().cpu().numpy(),
                         state=state.detach().cpu().numpy(),
                         loss= total_sup_eval_loss)
                checkpoint_e2e = {
                    "encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict(),
                    "likelihood_est": self.likelihood_est.state_dict(),
                    "particle_encoder": self.particle_encoder.state_dict(),
                    "transition_model": self.transition_model.state_dict(),
                    'encoder_optim': self.encoder_optim.state_dict(),
                    'decoder_optim': self.decoder_optim.state_dict(),
                    'likelihood_est_optim': self.likelihood_est_optim.state_dict(),
                    'particle_encoder_optim': self.particle_encoder_optim.state_dict(),
                    'transition_model_optim': self.transition_model_optim.state_dict(),
                    'cond_model': self.cond_model.state_dict(),
                    'cond_model_optim': self.cond_model_optim.state_dict(),
                    'nf_dyn': self.nf_dyn.state_dict(),
                    'nf_dyn_optim': self.nf_dyn_optim.state_dict(),
                    'cnf_measurement': self.cnf_measurement.state_dict(),
                    'cnf_measurement_optim': self.cnf_measurement_optim.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint_e2e, os.path.join('logs', run_id, "models", 'e2e_model_bestval_e2e.pth'))

    def load_model(self, file_name):
        ckpt_e2e = torch.load(file_name)
        self.encoder.load_state_dict(ckpt_e2e['encoder'])
        self.decoder.load_state_dict(ckpt_e2e['decoder'])
        self.likelihood_est.load_state_dict(ckpt_e2e['likelihood_est'])
        self.particle_encoder.load_state_dict(ckpt_e2e['particle_encoder'])
        self.transition_model.load_state_dict(ckpt_e2e['transition_model'])
        self.encoder_optim.load_state_dict(ckpt_e2e['encoder_optim'])
        self.decoder_optim.load_state_dict(ckpt_e2e['decoder_optim'])
        self.likelihood_est_optim.load_state_dict(ckpt_e2e['likelihood_est_optim'])
        self.particle_encoder_optim.load_state_dict(ckpt_e2e['particle_encoder_optim'])
        self.transition_model_optim.load_state_dict(ckpt_e2e['transition_model_optim'])

        self.cond_model.load_state_dict(ckpt_e2e['cond_model'])
        self.nf_dyn.load_state_dict(ckpt_e2e['nf_dyn'])
        self.cond_model_optim.load_state_dict(ckpt_e2e['cond_model_optim'])
        self.nf_dyn_optim.load_state_dict(ckpt_e2e['nf_dyn_optim'])
        self.cnf_measurement.load_state_dict(ckpt_e2e['cnf_measurement'])
        self.cnf_measurement_optim.load_state_dict(ckpt_e2e['cnf_measurement_optim'])

        epoch = ckpt_e2e['epoch']

        print(f'Load epcoh: {epoch}')

    def train_val(self, train_loader, valid_loader, run_id):
        params = self.param
        epoch_num = params.num_epochs

        dirs = ['result', 'model', 'checkpoint', 'logger']
        flags = [os.path.isdir(dir) for dir in dirs]
        for i, flag in enumerate(flags):
            if not flag:
                os.mkdir(dirs[i])

        logger = SummaryWriter('./logger')

        start_epoch = -1

        if params.resume:
            print('Resume training!')
            self.load_model('./model/e2e_model_bestval_e2e.pth')

        if params.pretrain_ae:#False
            print("Pretrain autoencoder model!")
            self.pretrain_ae(train_loader, valid_loader, start_epoch=start_epoch, epoch_num=300, logger=logger)

        if params.e2e_train:#True
            # end-to-end training
            print('End-to-end training!')
            self.e2e_train(train_loader, valid_loader, start_epoch=start_epoch, epoch_num=epoch_num, logger=logger, run_id = run_id)

    def testing(self, test_loader, run_id, model_path='./model/e2e_model_bestval_e2e.pth'):

        params = self.param
        if self.param.testing:
            print('Testing!')
            print('Load trained model')
            self.load_model(os.path.join(model_path, 'e2e_model_bestval_e2e.pth'))

        for epoch in range(1):
            # test
            self.set_eval_mode()
            total_sup_eval_loss = []

            with torch.no_grad():
                for iteration, inputs in enumerate(test_loader):

                    self.set_zero_grad()

                    loss_all, loss_sup, loss_pseud_lik, loss_ae, predictions, particle_list, particle_weight_list, state, start_state, image, likelihood_list, noise_list,obs_likelihood = self.forward(
                        inputs, train=False)
                    total_sup_eval_loss.append(loss_sup.detach().cpu().numpy())

            np.save(os.path.join('logs', run_id, "data", 'test_loss_epoch.npy'), total_sup_eval_loss)
            print(f"End-to-end loss testing: loss: {np.mean(total_sup_eval_loss)}")

            np.savez(os.path.join('logs', run_id, "data",'test_result.npz'),
                     particle_list= particle_list.detach().cpu().numpy(),
                     particle_weight_list=particle_weight_list.detach().cpu().numpy(),
                     likelihood_list=likelihood_list.detach().cpu().numpy(),
                     state=state.detach().cpu().numpy(),
                     pred=predictions.detach().cpu().numpy(),
                     images=image.detach().cpu().numpy(),
                     noise=noise_list.detach().cpu().numpy())
