import argparse
import os
import time
import importlib
from tqdm import tqdm
import numpy as np

import torch

from torch.utils.tensorboard import SummaryWriter
from FireCube_dataloader import FireCubeLoader
from utils import utils

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(BASE_DIR, 'models'))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def parse_args():

    parser = argparse.ArgumentParser('Trainer')
    parser.add_argument('--model', type=str, default='CNN', help='model name [default: CNN]')

    parser.add_argument('--batch_size', type=int, default=256, help='batch Size [default: 256]')
    parser.add_argument('--n_workers', type=int, default=8, help='number of workers [default: 8]')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory when using GPU [default: True]')
    parser.add_argument('--seed', type=int, default=0, help='random seed [default: 0]')
    parser.add_argument('--name', type=str, default='test_2021', help='name of the experiment [default test]')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU to use, use -1 for CPU [default: 0]')

    parser.add_argument('--test_year', type=int, default=2021, help='validation year [default: 2019]')
    parser.add_argument('--negative', type=str, default='clc',
                        help='whether to use clc or random to sample negative [default: clc]')
    parser.add_argument('--nan_fill', type=float, default=0., help='value to replace missing values [default: 0.]')
    parser.add_argument('--is_scale', type=bool, default=True, help='scale data [default: True]')
    parser.add_argument('--lag', default=10, type=int, help='number of days [default: 10]')
    parser.add_argument('--neg_pos_ratio_val', type=int, default=None,
                        help='ratio of negative to positive for validation [default: None]')

    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout ratio [default: 0.5]')
    parser.add_argument('--PE', type=bool, default=True, help='option to use positional encoding [default: True]')

    parser.add_argument('--model_dir',
                        type=str, default=r'./pretrained_models/pretrained_CNN_w_TE.pth', help='pretrained model')

    parser.add_argument('--data_dir', type=str,
                        default=r'/home/shams/Projects/FireCube/Dataset/datasets/datasets_grl/npy/spatiotemporal',
                        help='dataset path [default: None]')
    parser.add_argument('--log_dir', type=str, default=None, help='log name [default: None]')
    parser.add_argument('--dynamic_features', type=str, default=['1 km 16 days NDVI',
                                                                 'LST_Day_1km',
                                                                 'LST_Night_1km',
                                                                 'era5_max_d2m',
                                                                 'era5_max_t2m',
                                                                 'era5_max_sp',
                                                                 'era5_max_tp',
                                                                 'sminx',
                                                                 'era5_max_wind_speed',
                                                                 'era5_min_rh']
                        , help='dynamic features to use')

    parser.add_argument('--static_features', type=str, default=['dem_mean',
                                                                'slope_mean',
                                                                'roads_distance',
                                                                'waterway_distance',
                                                                'population_density']
                        , help='static features to use')

    parser.add_argument('--clc_features', type=str, default=['clc_' + str(c) for c in range(10)],
                        help='land cover classes to use')

    return parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def test(args):

    # get logger
    logger = utils.get_logger(args, mode='test')

    # fix random seed
    utils.fix_seed(args.seed)

    # dataloader
    utils.log_string(logger, "loading dataset ...")

    VAL_DATASET = FireCubeLoader(root=args.data_dir, mode='val', is_scale=args.is_scale, neg_pos_ratio=args.neg_pos_ratio_val,
                                 val_year=args.test_year, negative=args.negative, nan_fill=args.nan_fill,
                                 is_aug=False, is_shuffle=False, lag=args.lag, dynamic_features=args.dynamic_features,
                                 static_features=args.static_features, clc_features=args.clc_features, seed=args.seed)

    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 pin_memory=args.pin_memory,
                                                 num_workers=args.n_workers)

    utils.log_string(logger, "# testing samples: %d" % len(VAL_DATASET))

    # get models
    utils.log_string(logger, "\nloading the model ...")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.gpu_id != '-1':
        device = 'cuda'
    else:
        device = 'cpu'

    def import_class(name):
        module = importlib.import_module("models." + name)
        return getattr(module, name)

    if args.model == 'CNN':
        classifier = import_class(args.model)(input_channels_d=len(args.dynamic_features),
                                              input_channels_s=len(args.static_features),
                                              input_channels_c=len(args.clc_features),
                                              n_classes=2, drop_out=args.drop_out, pe=args.PE, device=device).to(device)
    elif args.model == 'SwinTransformer3D':
        # TODO add args
        classifier = import_class(args.model)(
            in_chans=len(args.dynamic_features)+len(args.static_features)+len(args.clc_features),
            n_classes=2).to(device)

    elif args.model == 'TimeSformer':
        # TODO add args
        classifier = import_class(args.model)(
            in_chans=len(args.dynamic_features) + len(args.static_features) + len(args.clc_features),
            n_classes=2).to(device)
    else:
        raise ValueError('Unexpected model name {}'.format(args.model))

    utils.log_string(logger, "model parameters: %d" % utils.count_parameters(classifier))

    # load trained model
    checkpoint = torch.load(args.model_dir, map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'], strict=True)

    utils.log_string(logger, 'testing on FireCube dataset ...\n')
    time.sleep(0.2)

    # initialize helper functions for evaluation
    eval_val = utils.evaluator(logger, 'Testing')

    # testing

    with torch.no_grad():

        classifier = classifier.eval()
        loss_sum = 0

        time.sleep(0.2)

        for i, (data_s, data_d, data_c, target, data_t) in tqdm(enumerate(valDataLoader), total=len(valDataLoader),
                                                                smoothing=0.9, postfix="  validation"):

            data_s, data_d, data_c, data_t, target = torch.Tensor(data_s).float().to(device), \
                                                     torch.Tensor(data_d).float().to(device), \
                                                     torch.Tensor(data_c).float().to(device), \
                                                     torch.Tensor(data_t).to(device), \
                                                     torch.Tensor(target).long().to(device)

            pred = classifier(data_s, data_c, data_d, data_t)

            pred_prob = torch.exp(pred).cpu().numpy()
            eval_val(pred_prob, target.cpu().numpy())

        mean_loss_val = loss_sum / float(len(valDataLoader))
        eval_val.get_results(mean_loss_val, np.nan)

        time.sleep(0.1)

    eval_val.reset()


if __name__ == '__main__':

    args = parse_args()
    test(args)

