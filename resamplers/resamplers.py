import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class resampler(nn.Module):
    def __init__(self, param):
        super().__init__()
        if param.resampler_type=='ot':
            self.kargs={'eps':param.epsilon, 'scaling':param.scaling, 'threshold':param.threshold,'max_iter':param.max_iter, 'device':device}
            self.resampling=resampler_ot
        elif param.resampler_type=='soft':
            self.kargs = {'num_resampled':param.num_particles,'index':True,'alpha':param.alpha, 'device':device}
            self.resampling=soft_resampler
    def forward(self, particles, particle_probs):
        particles_resampled, particle_probs_resampled, index_p = self.resampling(particles, particle_probs, **self.kargs)
        return particles_resampled, particle_probs_resampled, index_p


def soft_resampler(particles, particle_probs, alpha, num_resampled, index=True, device='cuda'):
    assert 0.0 < alpha <= 1.0
    batch_size = particles.shape[0]

    # normalize
    # particle_probs = particle_probs / particle_probs.sum(dim=-1, keepdim=True)
    uniform_probs = torch.ones((batch_size, num_resampled)).to(device) / num_resampled

    # build up sampling distribution q(s)
    if alpha < 1.0:
        # soft resampling
        q_probs = torch.stack((particle_probs * alpha, uniform_probs * (1.0 - alpha)), dim=-1).to(device)
        q_probs = q_probs.sum(dim=-1)
        q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)
        particle_probs = particle_probs / q_probs
    else:
        # hard resampling
        q_probs = particle_probs
        particle_probs = uniform_probs

    # sample particle indices according to q(s)

    basic_markers = torch.linspace(0.0, (num_resampled - 1.0) / num_resampled, num_resampled)
    random_offset = torch.FloatTensor(batch_size).uniform_(0.0, 1.0 / num_resampled)
    markers = random_offset[:, None] + basic_markers[None, :]  # shape: batch_size * num_resampled
    cum_probs = torch.cumsum(q_probs, axis=1).to(device)
    ## for resampling stability
    cum_probs[:, -1] = 1.0000
    markers = markers.to(device)
    marker_matching = markers[:, :, None] > cum_probs[:, None, :]
    samples = marker_matching.sum(axis=2).int()

    idx = samples + num_resampled * torch.arange(batch_size)[:, None].repeat((1, num_resampled)).to(device)
    particles_resampled = particles.view((batch_size * num_resampled, -1))[idx, :]

    particle_probs_resampled = particle_probs.view((batch_size * num_resampled,))[idx]
    particle_probs_resampled = particle_probs_resampled / particle_probs_resampled.sum(dim=-1, keepdim=True)
    if index==True:
        return particles_resampled, particle_probs_resampled, idx
    else:
        return particles_resampled, particle_probs_resampled

def resampler_ot(particles, weights, eps=0.1, scaling=0.75, threshold=1e-3, max_iter=100
                 ,device='cuda', flag=torch.tensor(True,requires_grad=False)):
    logw=weights.log()
    batch_size, num_particles, dimensions = particles.shape
    particles_resampled, particle_probs_resampled, particles_probs_log_resampled=OT_resampling(particles, logw=logw, eps=eps,
                         scaling=scaling, threshold=threshold, max_iter=max_iter,
                         n=particles.shape[1],device=device,flag=flag)
    index_p = (torch.arange(num_particles)+num_particles* torch.arange(batch_size)[:, None].repeat((1, num_particles))).type(torch.int64).to(device)
    return particles_resampled, particle_probs_resampled, index_p

def diameter(x, y):
    diameter_x = x.std(dim=1, unbiased=False).max(dim=-1)[0]
    diameter_y = y.std(dim=1, unbiased=False).max(dim=-1)[0]
    res = torch.maximum(diameter_x, diameter_y)
    return torch.where(res == 0., 1., res.double())


def cost(x, y):
    return squared_distances(x, y) / 2.


def squared_distances(x, y):
    return torch.cdist(x, y, p=2.0) ** 2


def max_min(x, y):
    max_max = torch.maximum(x.max(dim=1)[0].max(dim=1)[0], y.max(dim=1)[0].max(dim=1)[0])
    min_min = torch.minimum(x.max(dim=1)[0].min(dim=1)[0], y.min(dim=1)[0].min(dim=1)[0])

    return max_max - min_min


def softmin(epsilon, cost_matrix, f):
    """Implementation of softmin function
    :param epsilon: float
        regularisation parameter
    :param cost_matrix:
    :param f:
    :return:
    """
    n = cost_matrix.shape[1]
    b = cost_matrix.shape[0]

    f_ = f.reshape([b, 1, n])
    temp_val = f_ - cost_matrix / epsilon.reshape([-1, 1, 1])
    log_sum_exp = torch.logsumexp(temp_val, dim=2)
    res = -epsilon.reshape([-1, 1]) * log_sum_exp

    return res


def sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy, epsilon, particles_diameter, scaling,
                  threshold, max_iter, device='cuda'):
    batch_size = log_alpha.shape[0]
    continue_flag = torch.ones([batch_size], dtype=bool).to(device)
    epsilon_0 = particles_diameter ** 2
    scaling_factor = scaling ** 2

    a_y_init = softmin(epsilon_0, cost_yx, log_alpha)
    b_x_init = softmin(epsilon_0, cost_xy, log_beta)

    a_x_init = softmin(epsilon_0, cost_xx, log_alpha)
    b_y_init = softmin(epsilon_0, cost_yy, log_beta)

    def stop_condition(i, _a_y, _b_x, _a_x, _b_y, continue_, _running_epsilon):
        n_iter_cond = i < max_iter - 1
        return torch.logical_and(torch.tensor(n_iter_cond, dtype=bool).to(device),
                                 torch.all(continue_.bool()))

    def apply_one(a_y, b_x, a_x, b_y, continue_, running_epsilon):
        running_epsilon_ = running_epsilon.reshape([-1, 1])
        continue_reshaped = continue_.reshape([-1, 1])
        # TODO: Hopefully one day tensorflow controlflow will be lazy and not strict...
        at_y = torch.where(continue_reshaped, softmin(running_epsilon, cost_yx, log_alpha + b_x / running_epsilon_),
                           a_y)
        bt_x = torch.where(continue_reshaped, softmin(running_epsilon, cost_xy, log_beta + a_y / running_epsilon_), b_x)

        at_x = torch.where(continue_reshaped, softmin(running_epsilon, cost_xx, log_alpha + a_x / running_epsilon_),
                           a_x)
        bt_y = torch.where(continue_reshaped, softmin(running_epsilon, cost_yy, log_beta + b_y / running_epsilon_), b_y)

        a_y_new = (a_y + at_y) / 2
        b_x_new = (b_x + bt_x) / 2

        a_x_new = (a_x + at_x) / 2
        b_y_new = (b_y + bt_y) / 2

        a_y_diff = (torch.abs(a_y_new - a_y)).max(dim=1)[0]
        b_x_diff = (torch.abs(b_x_new - b_x)).max(dim=1)[0]

        local_continue = torch.logical_or(a_y_diff > threshold, b_x_diff > threshold)
        return a_y_new, b_x_new, a_x_new, b_y_new, local_continue

    def body(i, a_y, b_x, a_x, b_y, continue_, running_epsilon):
        new_a_y, new_b_x, new_a_x, new_b_y, local_continue = apply_one(a_y, b_x, a_x, b_y, continue_,
                                                                       running_epsilon)
        new_epsilon = torch.maximum(running_epsilon * scaling_factor, epsilon)
        global_continue = torch.logical_or(new_epsilon < running_epsilon, local_continue)

        return i + 1, new_a_y, new_b_x, new_a_x, new_b_y, global_continue, new_epsilon

    total_iter = 0
    converged_a_y, converged_b_x, converged_a_x, converged_b_y = a_y_init, b_x_init, a_x_init, b_y_init
    final_epsilon = epsilon_0

    while stop_condition(total_iter, converged_a_y, converged_b_x, converged_a_x, converged_b_y, continue_flag,
                         final_epsilon):
        total_iter, converged_a_y, converged_b_x, converged_a_x, converged_b_y, continue_flag, final_epsilon = body(
            total_iter, converged_a_y, converged_b_x, converged_a_x, converged_b_y, continue_flag, final_epsilon)

    converged_a_y, converged_b_x, converged_a_x, converged_b_y, = converged_a_y.detach().clone(), converged_b_x.detach().clone(), converged_a_x.detach().clone(), converged_b_y.detach().clone()
    epsilon_ = epsilon.reshape([-1, 1])

    final_a_y = softmin(epsilon, cost_yx, log_alpha + converged_b_x / epsilon_)
    final_b_x = softmin(epsilon, cost_xy, log_beta + converged_a_y / epsilon_)
    final_a_x = softmin(epsilon, cost_xx, log_alpha + converged_a_x / epsilon_)
    final_b_y = softmin(epsilon, cost_yy, log_beta + converged_b_y / epsilon_)
    return final_a_y, final_b_x, final_a_x, final_b_y, total_iter + 2


def sinkhorn_potentials(log_alpha, x, log_beta, y, epsilon, scaling, threshold, max_iter, device='cuda'):
    cost_xy = cost(x, y.detach().clone())
    cost_yx = cost(y, x.detach().clone())
    cost_xx = cost(x, x.detach().clone())
    cost_yy = cost(y, y.detach().clone())
    scale = max_min(x, y).detach().clone()
    a_y, b_x, a_x, b_y, total_iter = sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy, epsilon,
                                                   scale, scaling, threshold, max_iter, device=device)

    return a_y, b_x, a_x, b_y, total_iter


def transport_from_potentials(x, f, g, eps, logw, n, device='cuda'):
    float_n = n
    log_n = torch.log(float_n).to(device)

    cost_matrix = cost(x, x)

    fg = torch.unsqueeze(f, 2) + torch.unsqueeze(g, 1)  # fg = f + g.T
    temp = fg - cost_matrix
    temp = temp / eps

    temp = temp - torch.logsumexp(temp, dim=1, keepdims=True) + log_n

    # We "divide" the transport matrix by its col-wise sum to make sure that weights normalise to logw.
    temp = temp + torch.unsqueeze(logw, 1)
    transport_matrix = torch.exp(temp)

    return transport_matrix
def transport_function(x, logw, eps, scaling, threshold, max_iter, n, device='cuda'):
    eps = torch.tensor(eps, dtype=torch.float).to(device)
    float_n = torch.tensor(n, dtype=torch.float).to(device)
    log_n = torch.log(float_n).to(device)
    uniform_log_weight = -log_n * torch.ones_like(logw).to(device)
    dimension = torch.tensor(x.shape[-1]).to(device)

    centered_x = x - x.mean(dim=1, keepdim=True).detach().clone()
    diameter_value = diameter(x, x)

    scale = diameter_value.reshape([-1, 1, 1]) * torch.sqrt(dimension)
    scaled_x = centered_x / scale.detach().clone()
    alpha, beta, _, _, _ = sinkhorn_potentials(logw, scaled_x, uniform_log_weight, scaled_x, eps, scaling, threshold,
                                               max_iter, device=device)
    transport_matrix = transport_from_potentials(scaled_x, alpha, beta, eps, logw, float_n, device=device)

    return transport_matrix

def transport_grad(x_original, logw, eps, scaling, threshold, max_iter, n, device='cuda', grad_output=None):
    transport_matrix=transport_function(x_original, logw, eps, scaling, threshold, max_iter, n).requires_grad_()
    transport_matrix.backward(grad_output)
    return x_original.grad, logw.grad

class transport(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, logw, x_, logw_, transport_matrix_):
        ctx.save_for_backward(transport_matrix_, x_, logw_)
        return transport_matrix_.clone()  # grad

    @staticmethod
    def backward(ctx, d_transport):
        d_transport=torch.clamp(d_transport, -1., 1.)
        transport_matrix_, x_, logw_ = ctx.saved_tensors
        dx, dlogw = torch.autograd.grad(transport_matrix_, [x_, logw_], grad_outputs=d_transport, retain_graph=True)
        return dx, dlogw, None, None, None


def resample(tensor, new_tensor, flags):
    ndim = len(tensor.shape)
    shape = [-1] + [1] * (ndim - 1)
    return torch.where(torch.reshape(flags, shape), new_tensor.float(), tensor.float())


def apply_transport_matrix(particles, weights, log_weights, transport_matrix, flags):
    float_n_particles = torch.tensor(particles.shape[1]).float()
    transported_particles = torch.matmul(transport_matrix.float(), particles.float())
    uniform_log_weights = -float_n_particles.log() * torch.ones_like(log_weights)
    uniform_weights = torch.ones_like(weights) / float_n_particles

    resampled_particles = resample(particles, transported_particles, flags)
    resampled_weights = resample(weights, uniform_weights, flags)
    resampled_log_weights = resample(log_weights, uniform_log_weights, flags)

    return resampled_particles, resampled_weights, resampled_log_weights


def OT_resampling(x, logw, eps, scaling, threshold, max_iter, n, device='cuda',
                  flag=torch.tensor(True, requires_grad=False)):
    flag = flag.to(device)
    x_, logw_ = x.detach().clone().requires_grad_(), logw.detach().clone().requires_grad_()
    transport_matrix_ = transport_function(x_, logw_, eps, scaling, threshold, max_iter, n, device)

    calculate_transport=transport.apply
    transport_matrix = calculate_transport(x, logw, x_, logw_, transport_matrix_)
    resampled_particles, resampled_weights, resampled_log_weights = apply_transport_matrix(x, logw.exp(), logw,
                                                                                           transport_matrix, flag)
    return resampled_particles, resampled_weights, resampled_log_weights
