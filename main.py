import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import ToyDiskDataset
from arguments import parse_args
import random
from DPFs import DPF
import os
 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_run_id(args):
    cnt = '{}_NF^{}_{}_{}_{}_{}_{}_resample^{}_{}'.format(args.seed, args.NF_dyn, args.trainType, args.pos_noise, args.vel_noise, args.NF_lr, args.lr, args.resampler_type, args.measurement)
    return cnt
 
if __name__ == "__main__":
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    # configuration
    args = parse_args()
    seed=args.seed
    setup_seed(seed)
    print(args)
    run_id=get_run_id(args)

    logs_dir=os.path.join('logs', run_id)
    model_dir=os.path.join('logs', run_id, "models")
    data_dir=os.path.join('logs', run_id, "data")

    dirs=[logs_dir, model_dir, data_dir]
    flags=[os.path.isdir(dir) for dir in dirs]
    for i,flag in enumerate(flags):
        if not flag:
            os.mkdir(dirs[i])

    # task dataset
    Disk_Train = ToyDiskDataset(data_path='./data/disk/TwentyfiveDistractors/', filename='toy_pn={}_d=25_const'.format(args.true_pos_noise),
                                # threeDistractors_400, toy_pn=0.1_d=3_const; Skewt_15Distr_400
                                datatype="train_data")
    Disk_Val = ToyDiskDataset(data_path='./data/disk/TwentyfiveDistractors/', filename='toy_pn={}_d=25_const'.format(args.true_pos_noise),
                              # threeDistractors_400, toy_pn=0.1_d=3_const
                              datatype="val_data")
    train_loader = DataLoader(Disk_Train, batch_size=args.batchsize, shuffle=True, drop_last=True)
    valid_loader = DataLoader(Disk_Val, batch_size=50, shuffle=False, drop_last=True)

    dpf = DPF(args).to(device)
    if not args.testing:
        dpf.train_val(train_loader, valid_loader, run_id)

        torch.save(dpf, './model/dpf.pkl')

    Disk_Test = ToyDiskDataset(data_path='./data/disk/TwentyfiveDistractors/', filename='toy_pn={}_d=25_const'.format(args.true_pos_noise),
                               # threeDistractors_400, toy_pn=0.1_d=3_const#
                               datatype="test_data")
    test_loader = DataLoader(Disk_Test, batch_size=50, shuffle=False, drop_last=True)

    dpf.testing(test_loader, run_id=run_id, model_path=args.model_path)



