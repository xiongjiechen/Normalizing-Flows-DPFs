import torch
import torch.nn as nn


class NormalizingFlowModel(nn.Module):

    def __init__(self, prior, flows, device='cuda'):
        super().__init__()
        self.prior = prior
        self.device = device
        self.flows = nn.ModuleList(flows).to(self.device)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        # z, prior_logprob = x, self.prior.log_prob(x.float())
        z, prior_logprob = x, None
        return z, prior_logprob, log_det

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,)).to(self.device)
        x, _ = self.inverse(z)
        return x

class NormalizingFlowModel_cond(nn.Module):

    def __init__(self, prior, flows, device='cuda'):
        super().__init__()
        self.prior = prior
        self.device = device
        self.flows = nn.ModuleList(flows).to(self.device)

    def forward(self, x,obser):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x,obser)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z, obser):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z,obser)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples,obser):
        z = self.prior.sample((n_samples,)).to(self.device)
        x, _ = self.inverse(z,obser)
        return x

