import argparse
import gc
import os
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
# from diswotv2.helper.loops import validate
from diswotv2.helper.util import adjust_learning_rate
from diswotv2.losses import RMI_loss
from diswotv2.models import model_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_teacher(model_path, n_cls):
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))['model'])
    return model


def train_distiller(epoch,
                    train_loader,
                    module_list,
                    criterion_list,
                    optimizer,
                    opt,
                    offline=False,
                    total_epoch=None,
                    verbose=True):
    """One epoch distillation"""

    gamma = opt.gamma
    alpha = opt.alpha

    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_kd = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (image, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        # ===================forward=====================
        feat_s, k2, logit_s = model_s(image, is_feat=True)

        if not offline:
            with torch.no_grad():
                feat_t, k2, logit_t = model_t(image, is_feat=True)
                feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        # f3, f4, x
        loss_kd = criterion_kd(image, target, model_t, model_s)

        loss = gamma * loss_cls + loss_kd * alpha

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), image.size(0))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # gabage collection and release memory
        del feat_t
        gc.collect()
        torch.cuda.empty_cache()

        if i % 50 == 0 and verbose:
            print('Epoch: [{0}/{1}]\tStep: {2}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch,
                      total_epoch,
                      i,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, losses.avg


def validate_distiller(val_loader,
                       model_s,
                       model_t,
                       criterion,
                       opt,
                       loss_type='re_l1',
                       verbose=True):
    """validation"""
    assert loss_type in {'re_l1', 'rmi_loss'}
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # batch size is 64
    if loss_type == 're_l1':
        feat_loss = nn.MSELoss()  # L1
        feat_loss.cuda()
    else:
        feat_loss = RMI_loss(opt.batch_size)
        feat_loss.cuda()

    # switch to evaluate mode
    model_s.eval()

    with torch.no_grad():
        # end = time.time()
        for _, (image, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()

            # compute output
            feat_t, logit_t = model_t(image, is_feat=True)
            feat_s, logit_s = model_s(image, is_feat=True)

            loss_logits = criterion(logit_s, target)

            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if loss_type == 're_l1':
                feat_t_var = feat_t[-2].var()
                loss_feats = feat_loss(feat_s[-2], feat_t[-2]) / (
                    feat_t_var + 1e-9)
            elif loss_type == 'rmi_loss':
                loss_feats = feat_loss(feat_s[-2], feat_t[-2])

        if verbose:
            print('Evalidate:\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      loss=losses, top1=top1, top5=top5))
        loss = 0.5 * loss_feats + (1 - 0.5) * loss_logits
        losses.update(loss.item())

        print(f'Current Loss: {losses.avg:.3f} top1 Acc: {top1.avg:.4f}')

    return losses.avg


class diswotv2Distiller:
    """Mento calo distillation
    """

    def __init__(self,
                 opt,
                 epochs=2,
                 death_rate=0.5,
                 death_mode='magnitude',
                 growth='random',
                 redistribution='none',
                 density=0.05,
                 verbose=True):
        super().__init__()
        self.opt = opt
        self.epochs = epochs

        # sparse settings
        self.death_rate = 0.5 if death_rate is None else death_rate
        self.death_mode = 'magnitude' if death_mode is None else death_mode
        self.growth = 'random' if growth is None else growth
        self.redistribution = 'none' if redistribution is None else redistribution
        self.density = 0.05 if density is None else density
        self.verbose = verbose

        # asserations
        assert self.death_mode in {'magnitude', 'SET', 'threshold'}, \
            f'Invalid death mode: {self.death_mode}'
        assert self.growth in {'momentum', 'random', 'random_unfired', 'gradient'}, \
            f'Invalid growth mode: {self.growth}'
        assert self.redistribution in {'momentum', 'magnitude', 'nonzeros', 'none'}, \
            f'Invalid redistribution mode: {self.redistribution}'

        self.setUp()

    def setUp(self) -> None:
        # tensorboard logger
        self.logger = tb_logger.Logger(
            logdir=self.opt.log_folder, flush_secs=2)

        if self.opt.dataset != 'cifar100':
            raise NotImplementedError(self.opt.dataset)

        self.train_loader, self.val_loader = \
            get_cifar100_dataloaders(data_folder='./data',
                                     batch_size=128,
                                     num_workers=0)

        n_cls = 100
        # model
        self.model_t = load_teacher(self.opt.path_t, n_cls)
        self.model_s = model_dict[self.opt.model_s](num_classes=n_cls)

        self.module_list = nn.ModuleList([])
        self.module_list.append(self.model_s)
        self.module_list.append(self.model_t)

        self.criterion_list = []
        self.criterion_list.append(nn.CrossEntropyLoss())
        self.optimizer = optim.SGD(
            self.module_list.parameters(),
            lr=self.opt.learning_rate,
            momentum=self.opt.momentum,
            weight_decay=self.opt.weight_decay)
        if torch.cuda.is_available():
            self.module_list.cuda()
            cudnn.benchmark = True

    def estimate_rewards(self, loss_fn) -> float:
        self.criterion_list.append(loss_fn)

        if torch.cuda.is_available():
            self.criterion_list[0].cuda()

        for epoch in range(1, self.epochs + 1):
            adjust_learning_rate(epoch, self.opt, self.optimizer)

            train_distiller(
                epoch,
                train_loader=self.train_loader,
                module_list=self.module_list,
                criterion_list=self.criterion_list,
                optimizer=self.optimizer,
                opt=self.opt,
                total_epoch=self.epochs,
                verbose=self.verbose)

        # loss type is `re_l1` or `rmi_loss`
        return validate_distiller(
            self.val_loader,
            self.model_s,
            self.model_t,
            self.criterion_list[0],
            self.opt,
            loss_type='re_l1',
            verbose=self.verbose)


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return f'{segments[0]}_{segments[1]}_{segments[2]}'


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument(
        '--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument(
        '--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument(
        '--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument(
        '--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument(
        '--epochs', type=int, default=24, help='number of training epochs')
    parser.add_argument(
        '--init_epochs',
        type=int,
        default=30,
        help='init training for two-stage methods')

    # optimization
    parser.add_argument(
        '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument(
        '--lr_decay_epochs',
        type=str,
        default='10,20',
        help='where to decay lr, can be a list')
    parser.add_argument(
        '--lr_decay_rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument(
        '--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar100',
        choices=['cifar100'],
        help='dataset')

    # model
    parser.add_argument(
        '--model_s',
        type=str,
        default='resnet20',
        choices=[
            'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44',
            'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4', 'wrn_16_1',
            'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'vgg8', 'vgg11', 'vgg13',
            'vgg16', 'vgg19', 'ResNet50', 'MobileNetV2', 'ShuffleV1',
            'ShuffleV2'
        ])
    parser.add_argument(
        '--path_t',
        type=str,
        default='./save/models/resnet110_vanilla/ckpt_epoch_240.pth',
        help='teacher model snapshot')

    # distillation
    parser.add_argument(
        '--distill',
        type=str,
        default='kd',
        choices=[
            'kd', 'hint', 'attention', 'similarity', 'correlation', 'vid',
            'crd', 'kdsvd', 'fsp', 'rkd', 'pkt', 'abound', 'factor', 'nst',
            'spatial_kl', 'channel_kl', 'channel_gmml2', 'batch_kl'
        ])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument(
        '-r',
        '--gamma',
        type=float,
        default=1,
        help='weight for classification')
    parser.add_argument(
        '-a', '--alpha', type=float, default=1, help='weight balance for KD')
    parser.add_argument(
        '-b',
        '--beta',
        type=float,
        default=1,
        help='weight balance for other losses')

    # KL distillation
    parser.add_argument(
        '--kd_T',
        type=float,
        default=4,
        help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument(
        '--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument(
        '--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument(
        '--nce_k',
        default=16384,
        type=int,
        help='number of negative samples for NCE')
    parser.add_argument(
        '--nce_t',
        default=0.07,
        type=float,
        help='temperature parameter for softmax')
    parser.add_argument(
        '--nce_m',
        default=0.5,
        type=float,
        help='momentum for non-parametric updates')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S-{}_T-{}_{}_{}_r-{}_a-{}_b-{}_{}'.format(
        opt.model_s, opt.model_t, opt.dataset, opt.distill, opt.gamma,
        opt.alpha, opt.beta, opt.trial)

    opt.log_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    from diswotv2.searchspace.interactions import ParaInteraction
    opt = parse_option()
    kz_distiller = diswotv2Distiller(opt)
    loss_fn = ParaInteraction()
    kz_distiller.estimate_rewards(loss_fn)


if __name__ == '__main__':
    main()
