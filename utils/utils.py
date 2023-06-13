import torch
import numpy as np
import random

import os

import datetime
import logging
from torchmetrics import AUROC


def log_string(logger, str):
    logger.info(str)
    print(str)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)


def get_logger(args, mode='train'):

    if args.name is None:
        args.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    if args.log_dir is None:
        args.log_dir = './log'

    log_dir = os.path.join(args.log_dir, args.name)
    make_dir(log_dir)

    if mode == 'train':
        checkpoints_dir = os.path.join(log_dir, 'model_checkpoints/')
        make_dir(checkpoints_dir)

    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log_file.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string(logger, 'Parameters ...')
    log_string(logger, args)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_optimizer(optim_groups, optim, lr, weight_decay):

    if optim == 'Adam':
        optimizer = torch.optim.Adam(optim_groups, lr=lr, weight_decay=weight_decay, eps=1e-04)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, weight_decay=weight_decay)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(optim_groups, lr=lr, weight_decay=weight_decay)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(optim_groups, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Unexpected optimizer {}'.format(optim))

    return optimizer


def get_learning_scheduler(optimizer, lr_scheduler, lr_step_size, lr_decay):

    if lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_decay)
    else:
        raise ValueError('Unexpected learning_scheduler {}'.format(lr_scheduler))

    return lr_scheduler



class evaluator():
    def __init__(self, logger, mode):

        self.classes = ['Background', 'Wildfire']
        self.n_classes = len(self.classes)

        self.mode = mode
        self.logger = logger

        self.correct_all = 0
        self.accuracy_all = 0
        self.seen_all = 0
        self.AUROC = 0

        self.accuracy = [0 for _ in range(self.n_classes)]
        self.precision = [0 for _ in range(self.n_classes)]
        self.F1 = [0 for _ in range(self.n_classes)]
        self.iou = [0 for _ in range(self.n_classes)]

        self.weights_label = np.zeros(self.n_classes)
        self.seen_label_all = [0 for _ in range(self.n_classes)]
        self.correct_label_all = [0 for _ in range(self.n_classes)]
        self.iou_de_label_all = [0 for _ in range(self.n_classes)]
        self.predicted_label_all = [0 for _ in range(self.n_classes)]

        self.preds_prob = []
        self.targets = []

        self.AUROC_metric = AUROC(num_classes=2, pos_label=1, task='binary')

    def get_results(self, mean_loss, best_loss):

        weights_label = self.weights_label.astype(np.float32) / np.sum(self.weights_label.astype(np.float32))
        self.accuracy_all = self.correct_all / float(self.seen_all)

        message = '-----------------   %s   -----------------\n' % self.mode

        for label in range(self.n_classes):
            self.precision[label] = self.correct_label_all[label] / float(self.predicted_label_all[label])
            self.accuracy[label] = self.correct_label_all[label] / (np.array(self.seen_label_all[label], dtype=float) + 1e-6)
            self.F1[label] = 2 * self.precision[label] * self.accuracy[label] / (self.accuracy[label] + self.precision[label])
            self.iou[label] = self.correct_label_all[label] / float(self.iou_de_label_all[label])

        for label in range(self.n_classes):
            message += 'class %s weight: %.4f, precision: %.4f, accuracy: %.4f, F1: %.4f IoU: %.4f \n' % (
                self.classes[label] + ' '*(14 - len(self.classes[label])), weights_label[label],
                self.precision[label],
                self.accuracy[label],
                self.F1[label],
                self.iou[label])

        self.AUROC = self.AUROC_metric(torch.tensor(self.preds_prob, dtype=torch.float32),
                                      torch.tensor(self.targets, dtype=torch.int32))

        message += '\n%s accuracy      : %.4f' % (self.mode, self.accuracy_all)
        message += '\n%s mean accuracy : %.4f' % (self.mode, np.mean(self.accuracy))
        message += '\n%s AUROC         : %.4f' % (self.mode, self.AUROC)
        message += '\n%s mean IoU      : %.4f' % (self.mode, np.mean(self.iou))
        message += '\n%s mean F1       : %.4f' % (self.mode, np.mean(self.F1))
        message += '\n%s mean loss     : %.4f    best mean loss     : %.4f' % (self.mode, mean_loss, best_loss)

        log_string(self.logger, message)


    def reset(self):

        self.correct_all = 0
        self.accuracy_all = 0
        self.seen_all = 0
        self.AUROC = 0

        self.accuracy = [0 for _ in range(self.n_classes)]
        self.precision = [0 for _ in range(self.n_classes)]
        self.F1 = [0 for _ in range(self.n_classes)]
        self.iou = [0 for _ in range(self.n_classes)]

        self.weights_label = np.zeros(self.n_classes)
        self.seen_label_all = [0 for _ in range(self.n_classes)]
        self.correct_label_all = [0 for _ in range(self.n_classes)]
        self.iou_de_label_all = [0 for _ in range(self.n_classes)]
        self.predicted_label_all = [0 for _ in range(self.n_classes)]

        self.preds_prob = []
        self.targets = []

    def __call__(self, pred_prob, target):

        pred = np.argmax(pred_prob, axis=1)

        correct = np.sum(pred == target)
        self.correct_all += correct
        self.seen_all += len(target)

        weights, _ = np.histogram(target, range(self.n_classes + 1))
        self.weights_label += weights

        for label in range(self.n_classes):
            self.correct_label_all[label] += np.sum((pred == label) & (target == label))
            self.seen_label_all[label] += np.sum((target == label))
            self.iou_de_label_all[label] += np.sum(((pred == label) | (target == label)))
            self.predicted_label_all[label] += np.sum(pred == label)

        self.preds_prob.extend(pred_prob[:, 1].flatten())
        self.targets.extend(target.flatten())



def save_model(model, epoch, mean_loss_train, mean_loss_val, logger, args, save_name):

    log_dir = os.path.join(args.log_dir, args.name)
    checkpoints_dir = os.path.join(log_dir, 'model_checkpoints/')
    path = os.path.join(checkpoints_dir, save_name)
    log_string(logger, 'saving model to %s' % path)

    state = {
        'epoch': epoch,
        'mean_loss_train': mean_loss_train,
        'mean_loss_validation': mean_loss_val,
        'model_state_dict': model.state_dict(),
    }

    torch.save(state, path)

