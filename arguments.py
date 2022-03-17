# import configargparse
import argparse
import copy

def parse_args(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_false', help='whether to use GPU')
    parser.add_argument('--gpu-index', type=int, default=0, help='index num of GPU to use')
    parser.add_argument('--trainType', dest='trainType', type=str,
                        default='DPF', choices=['DPF', 'SDPF', 'UDPF'],
                        help='train type: supervised, semi, unsupervised learning')
    parser.add_argument('--pretrain_ae', action='store_true',
                        help='pretrain of autoencoder model')
    parser.add_argument('--pretrain-NFcond', action='store_true',
                        help='pretrain of conditional normalising flow model')
    parser.add_argument('--e2e-train', action='store_false',
                        help='End to end training')
    parser.add_argument('--load-pretrainModel', action='store_true',
                        help='Load pretrain model')

    parser.add_argument('--NF-dyn', action='store_true',help='train using normalising flow')
    parser.add_argument('--NF-cond', action='store_true',help='train using conditional normalising flow')
    parser.add_argument('--measurement',type=str, default='cos', help='|CRNVP|cos|NN|CGLOW|gaussian|')
    parser.add_argument('--NF-lr', type=float, default=2.5,help='NF learning rate')

    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon in OT resampling')
    parser.add_argument('--scaling', type=float, default=0.75, help='scaling in OT resampling')
    parser.add_argument('--alpha', type=float, default=0.5, help='hyperparameter for soft resampling')
    parser.add_argument('--threshold', type=float, default=1e-3, help='threshold in OT resampling')
    parser.add_argument('--max_iter', type=int, default=100, help='max iterarion in OT resampling')
    parser.add_argument('--resampler_type',type=str, default='ot', help='|ot|soft|')

    parser.add_argument('--resume', action='store_true',
                        help='resume training from checkpoint')

    parser.add_argument('--Dyn_nn', action='store_true',
                        help='learned dynamic model using neural network')
    parser.add_argument('--Obs_feature', action='store_false',
                        help='Compute likelihood using feature similarity')

    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--hiddensize', type=int, default=32, help='hidden size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('--optim', type=str, default='Adam',
                        help='type of optim')
    parser.add_argument('--num-epochs', type=int, default=150, help='num epochs')
    parser.add_argument('--num-particles', type=int, default=100, help='num of particles')

    parser.add_argument('--split-ratio', type=float, default=0.9, help='split training data')
    parser.add_argument('--labeledRatio', type=float, default=1.0, help='labeled training data')
    parser.add_argument('--init-with-true-state', action='store_true',
                        help='init_with_true_state, default: false, uniform initialisation')

    parser.add_argument('--dropout-keep-ratio', type=float, default=0.3, help='1-dropout_ratio')
    parser.add_argument('--particle_std', type=float, default=0.2, help='particle std')
    parser.add_argument('--seed', type=int, default=2, help='random seed')

    parser.add_argument('--sequence-length', dest='sequence_length', type=int,
                        default=50, help='length of the generated sequences')
    parser.add_argument('--width', dest='width', type=int, default=128,
                        help='width (= height) of the generated observations')

    parser.add_argument('--pos-noise', dest='pos_noise', type=float,
                        default=20.0,
                        help='sigma for the positional process noise')
    parser.add_argument('--vel-noise', dest='vel_noise', type=float,
                        default=20.0,
                        help='sigma for the velocity noise')

    parser.add_argument('--true-pos-noise', dest='true_pos_noise', type=float,
                        default=2.0,
                        help='sigma for the positional process noise when generating datasets')
    parser.add_argument('--true-vel-noise', dest='true_vel_noise', type=float,
                        default=2.0,
                        help='sigma for the velocity noise when generating datasets')

    parser.add_argument('--block-length', dest='block_length', type=int,
                        default=10, help='block length for pseudo-likelihood')

    parser.add_argument('--testing', action='store_true',
                        help='Check testing performance')
    parser.add_argument('--model-path', type=str, default='./model/e2e_model_bestval_e2e.pth', help='path of saved model')


    parser.add_argument("--x_size", type=tuple, default=(3,8,8))
    parser.add_argument("--y_size", type=tuple, default=(3,8,8))
    parser.add_argument("--x_hidden_channels", type=int, default=8)
    parser.add_argument("--x_hidden_size", type=int, default=16)
    parser.add_argument("--y_hidden_channels", type=int, default=8)
    parser.add_argument("-K", "--flow_depth", type=int, default=1)
    parser.add_argument("-L", "--num_levels", type=int, default=1)
    parser.add_argument("--learn_top", type=bool, default=False)

    parser.add_argument("--x_bins", type=float, default=256.0)
    parser.add_argument("--y_bins", type=float, default=256.0)

    parser.add_argument("--individual", action='store_true',help='set individual opimizers for different units')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    param = copy.deepcopy(args)
    for labeledRatio in param.labeledRatio:
        args.labeledRatio = labeledRatio
        print(args)

