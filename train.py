import argparse
import os
import time
import importlib
from tqdm import tqdm
import numpy as np

import torch

from torch.utils.tensorboard import SummaryWriter
from FireCube_dataloader import FireCubeLoader
from models.loss import NLLLoss
from utils import utils

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

#torch.autograd.set_detect_anomaly(True)

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(BASE_DIR, 'models'))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def parse_args():

    parser = argparse.ArgumentParser('Trainer')
    parser.add_argument('--model', type=str, default='CNN', help='model name [default: CNN]')

    parser.add_argument('--batch_size', type=int, default=256, help='batch Size [default: 256]')
    parser.add_argument('--n_workers', type=int, default=10, help='number of workers [default: 8]')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory when using GPU [default: True]')
    parser.add_argument('--seed', type=int, default=0, help='random seed [default: 0]')
    parser.add_argument('--name', type=str, default='test', help='name of the experiment [default test]')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU to use, use -1 for CPU [default: 0]')

    parser.add_argument('--val_year', type=int, default=2019, help='validation year [default: 2019]')
    parser.add_argument('--negative', type=str, default='clc',
                        help='whether to use clc or random to sample negative [default: clc]')
    parser.add_argument('--nan_fill', type=float, default=0., help='value to replace missing values [default: 0.]')
    parser.add_argument('--is_aug', type=bool, default=False, help='data augmentation [default: False]')
    parser.add_argument('--is_scale', type=bool, default=True, help='scale data [default: True]')
    parser.add_argument('--is_shuffle', type=bool, default=True, help='shuffle data [default: True]')
    parser.add_argument('--lag', default=10, type=int, help='number of days [default: 10]')
    parser.add_argument('--neg_pos_ratio_train', type=int, default=2,
                        help='ratio of negative to positive for training [default: 2]')
    parser.add_argument('--neg_pos_ratio_val', type=int, default=None,
                        help='ratio of negative to positive for validation [default: None]')

    parser.add_argument('--n_epochs', default=40, type=int, help='number of epochs [default: 40]')
    parser.add_argument('--lr', default=0.00003, type=float, help='initial learning rate [default: 0.00003]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam [default: Adam]')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay [default: 2e-2]')

    parser.add_argument('--drop_out', type=float, default=0.5, help='dropout ratio [default: 0.5]')
    parser.add_argument('--PE', type=bool, default=True, help='option to use positional encoding [default: True]')

    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=23, help='learning rate step decay')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--e_save', type=int, default=2, help='save model every x epochs [default: 2]')

    parser.add_argument('--data_dir', type=str,
                        default=r'./datasets/datasets_grl/npy/spatiotemporal',
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

def train(args):

    # get logger
    logger = utils.get_logger(args, mode='train')

    # get tensorboard writer
    writer = SummaryWriter(os.path.join(args.log_dir, args.name))

    # fix random seed
    utils.fix_seed(args.seed)

    # dataloader
    utils.log_string(logger, "loading dataset ...")

    TRAIN_DATASET = FireCubeLoader(root=args.data_dir, mode='train', is_scale=args.is_scale, neg_pos_ratio=args.neg_pos_ratio_train,
                                   val_year=args.val_year, negative=args.negative, nan_fill=args.nan_fill,
                                   is_aug=args.is_aug, is_shuffle=args.is_shuffle, lag=args.lag, dynamic_features=args.dynamic_features,
                                   static_features=args.static_features, clc_features=args.clc_features, seed=args.seed)

    VAL_DATASET = FireCubeLoader(root=args.data_dir, mode='val', is_scale=args.is_scale, neg_pos_ratio=args.neg_pos_ratio_val,
                                 val_year=args.val_year, negative=args.negative, nan_fill=args.nan_fill,
                                 is_aug=False, is_shuffle=False, lag=args.lag, dynamic_features=args.dynamic_features,
                                 static_features=args.static_features, clc_features=args.clc_features, seed=args.seed)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET,
                                                  batch_size=args.batch_size,
                                                  shuffle=args.is_shuffle,
                                                  pin_memory=args.pin_memory,
                                                  num_workers=args.n_workers)

    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 pin_memory=args.pin_memory,
                                                 num_workers=args.n_workers)

    utils.log_string(logger, "# training samples: %d" % len(TRAIN_DATASET))
    utils.log_string(logger, "# evaluation samples: %d" % len(VAL_DATASET))

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

    # get losses
    utils.log_string(logger, "get criterion ...")
    class_weights = torch.Tensor([0.5, 0.5]).to(device)
    criterion = NLLLoss(weight=class_weights).to(device)

    # get optimizer
    utils.log_string(logger, "get optimizer and learning rate scheduler ...")
    optimizer = utils.get_optimizer(classifier.parameters(), args.optimizer, args.lr, args.weight_decay)
    lr_scheduler = utils.get_learning_scheduler(optimizer, args.lr_scheduler, args.lr_step_size, args.lr_decay)

    utils.log_string(logger, 'training on FireCube dataset ...\n')
    time.sleep(0.2)

    # initialize the best values
    best_loss_train = np.inf
    best_loss_val = np.inf

    # initialize helper functions for evaluation
    eval_train = utils.evaluator(logger, 'Training')
    eval_val = utils.evaluator(logger, 'Validation')

    # training and evaluation loop

    for epoch in range(args.n_epochs):
        utils.log_string(logger, '################# Epoch (%s/%s) #################' % (epoch + 1, args.n_epochs))

        # training
        classifier = classifier.train()
        loss_sum = 0

        time.sleep(0.2)

        for i, (data_s, data_d, data_c, target, data_t) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader),
                                                                smoothing=0.9, postfix="  training"):

            optimizer.zero_grad()

            data_s, data_d, data_c, data_t, target = torch.Tensor(data_s).float().to(device), \
                                                     torch.Tensor(data_d).float().to(device),\
                                                     torch.Tensor(data_c).float().to(device),\
                                                     torch.Tensor(data_t).to(device), \
                                                     torch.Tensor(target).long().to(device)

            #data_m = torch.Tensor(data_m).long().to(device)

            pred = classifier(data_s, data_c, data_d, data_t)

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            """
            if i % 5 == 0:
                clipping_value = 5  # arbitrary value of your choosing
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), clipping_value)
                total_norm = 0
                parameters = [p for p in model.parameters() if p.requires_grad]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar("Grad_L2_Norm", total_norm, i)
            """
            pred_prob = torch.exp(pred.detach()).cpu().numpy()
            eval_train(pred_prob, target.detach().cpu().numpy())

        mean_loss_train = loss_sum / float(len(trainDataLoader))
        eval_train.get_results(mean_loss_train, best_loss_train)

        if mean_loss_train <= best_loss_train:
            best_loss_train = mean_loss_train

        time.sleep(0.1)

        # validating

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

                loss = criterion(pred, target)
                loss_sum += loss

                pred_prob = torch.exp(pred).cpu().numpy()
                eval_val(pred_prob, target.cpu().numpy())

            mean_loss_val = loss_sum / float(len(valDataLoader))
            eval_val.get_results(mean_loss_val, best_loss_val)

            if mean_loss_val <= best_loss_val:
                best_loss_val = mean_loss_val
                utils.save_model(classifier, epoch, mean_loss_train, mean_loss_val, logger, args, 'best_loss_model.pth')

            time.sleep(0.1)

        # save model every e_save epochs
        if epoch % args.e_save == 0:
            utils.save_model(classifier, epoch, mean_loss_train, mean_loss_val, logger, args, 'epoch_{}_model.pth'.format(epoch+1))

        # write curves to tensorboard
        writer.add_scalars("Loss", {'train': mean_loss_train, 'val': mean_loss_val}, epoch+1)
        writer.add_scalars("Acc", {'train': eval_train.accuracy_all, 'val': eval_val.accuracy_all}, epoch+1)
        writer.add_scalars("Accuracy_Positive", {'train': eval_train.accuracy[1],
                                                 'val': eval_val.accuracy[1]}, epoch+1)
        writer.add_scalars("Precision_Positive", {'train': eval_train.precision[1],
                                                  'val': eval_val.precision[1]}, epoch+1)
        writer.add_scalars("F1_Positive", {'train': eval_train.F1[1],
                                           'val': eval_val.F1[1]}, epoch+1)
        writer.add_scalars("AUROC", {'train': eval_train.AUROC,
                                     'val': eval_val.AUROC}, epoch+1)

        eval_train.reset()
        eval_val.reset()

        lr_scheduler.step()


if __name__ == '__main__':

    args = parse_args()
    train(args)

