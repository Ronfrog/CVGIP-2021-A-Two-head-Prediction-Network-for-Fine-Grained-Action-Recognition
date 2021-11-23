import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import os
import numpy as np
import math

from config import get_args
from datasets import BasketballFoulsDataset
from utils.filehandler import saveTorchModel, loadTorchModel
from models import resnet, resnet2d1d, wide_resnet, i3d, ours
from utils.logger import MyLogger
from utils.lossfunction import TwoLinesLoss

torch.cuda.set_device(1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]

def adjust_lr(iteration, optimizer, schedule):
    for param_group in optimizer.param_groups:
        param_group["lr"] = schedule[iteration]

def build_logger(args):
    train_logger = MyLogger(args.trainlog_root, "Training Log : Basketball Foul Detection v2.")
    eval_logger = MyLogger(args.evallog_root, "Evaluation Log : Basketball Foul Detection v2.")
    return train_logger, eval_logger

def build_dataloader(args):
    trainset = BasketballFoulsDataset(istrain = True,
                                      data_paths_json = args.train_data_json,
                                      videos_root = args.train_data_root, 
                                      data_size = args.data_size,
                                      select_type = args.select_type,
                                      select_freq = args.select_freq,
                                      select_random_start = args.select_random_start,
                                      select_random_interval = args.select_random_interval)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    evalset = BasketballFoulsDataset(istrain = False,
                                     data_paths_json = args.eval_data_json,
                                     videos_root = args.eval_data_root, 
                                     data_size = args.data_size,
                                     select_type = args.select_type,
                                     select_freq = args.select_freq,
                                     select_random_start = False,
                                     select_random_interval = 0)
    eval_loader = torch.utils.data.DataLoader(evalset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_loader, eval_loader


def build_model(args):
    if args.model_name == "ours":
        model = ours.ScoresModel(args.data_size[0], args.num_scores, args.num_classes)
    elif args.model_name == "resnet":
        model = resnet.generate_model(50, n_classes=args.num_classes)
    elif args.model_name == "resnet2+1d":
        model = resnet2d1d.generate_model(50, n_classes=args.num_classes)
    elif args.model_name == "i3d":
        model = i3d.InceptionI3d(num_classes=args.num_classes)
    elif args.model_name == "wide_resnet":
        model = wide_resnet.generate_model(50, n_classes=args.num_classes)
    elif args.model_name == "ours-resnet":
        model = resnet.TwoHeads_ResNet(resnet.Bottleneck, [3, 4, 6, 3], resnet.get_inplanes(), n_classes=args.num_classes)
    elif args.model_name == "ours-i3d":
        model = i3d.TwoHeads_InceptionI3d(num_classes=args.num_classes)
    return model



def build_optimizer(model, dataset_size, args):
    """
    dataset_size: batchs number in one epoch.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr
    )

    warmup_lr_schedule = np.linspace(1e-6, args.base_lr, dataset_size * args.warmup_epochs)
    iters = np.arange(dataset_size * (args.max_epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([0.5 * args.base_lr * (1 + \
                         math.cos(math.pi * t / (dataset_size * (args.max_epochs - args.warmup_epochs)))) for t in iters])
    schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    
    return optimizer, schedule

def build_criterion(args):
    criterion = TwoLinesLoss(use_focal_loss=args.use_focal_loss,
                             use_info_gain=args.use_info_gain)
    cross_entropy = nn.CrossEntropyLoss()
    return criterion, cross_entropy


def train(epoch, model, loader, optimizer, schedule, criterion, ce_loss, train_logger, args):

    train_logger.set_epoch(epoch + 1)

    train_logger.log("\ntrain epoch : {}".format(epoch + 1))

    model.train()

    end_time = time.time()
    loss_infos = [0, 0, 0]
    optimizer.zero_grad()
    corrects, total = 0, 0
    for batch_id, (datas, labels, pair_datas, pair_labels) in enumerate(loader):

        # measure data loading time
        data_time = time.time() - end_time

        # adjust learning
        iteration = epoch * len(loader) + batch_id
        # adjust_lr(iteration, optimizer, schedule)

        # forward
        datas, labels = datas.cuda(1), labels.cuda(1)
        pair_datas, pair_labels = pair_datas.cuda(1), pair_labels.cuda(1)

        if "ours" in args.model_name:
            c_outs, s_outs = model(datas)
            loss = ce_loss(c_outs, labels)
            c_outs2, s_outs2 = model(pair_datas)
            loss2, loss_infos = criterion(s_outs, labels, s_outs2, pair_labels)
            loss += loss2
        else:
            c_outs = model(datas)
            loss = ce_loss(c_outs, labels) # compute loss
            c_outs2 = model(pair_datas)
            loss += ce_loss(c_outs2, pair_labels) # compute loss

        _, preds = torch.max(c_outs, 1)
        # use tmp for update logger
        tmp_corrects = (preds == labels).sum()
        tmp_total = labels.size(0)
        # measure top1 accuracy.
        corrects += tmp_corrects
        total += tmp_total

        # backward
        loss = loss/args.backward_freq
        loss.backward()

        if (batch_id + 1) % args.backward_freq == 0:
            optimizer.step()
            optimizer.zero_grad()

        forward_time = time.time() - end_time 

        batch_size = labels.size(0)

        end_time = time.time()
        # update loagger information.
        train_logger.update(batch_id+1, 
                            data_time, 
                            forward_time, 
                            loss.item(), 
                            loss_infos[0] + loss_infos[1],
                            loss_infos[2],
                            tmp_corrects/tmp_total, 
                            0,
                            0,
                            batch_size, 
                            get_lr(optimizer))

        # ============= misc =====================
        if (batch_id + 1) % args.log_freq == 0:
            train_logger.show()

    train_logger.log("\n\n Epochs {} Training Accuracy:{}.\n\n".format(epoch+1, corrects/total))

@torch.no_grad()
def eval(epoch, model, loader, eval_logger, args):

    eval_logger.set_epoch(epoch + 1)
    eval_logger.log("\nstart evaluate after {} epoch unsupervised training.\n".format(epoch + 1))

    # change to evaluation mode.
    model.eval()
    corrects, total = 0, 0
    nf_info_gain, f_info_gain = 0, 0
    end_time = time.time()
    for batch_id, (datas, labels) in enumerate(loader):

        # measure data loading time
        data_time = time.time() - end_time
        batch_size = labels.size(0)
        datas = datas.cuda(1)
        if "ours" in args.model_name:
            c_outs, s_outs = model(datas)
            c_outs = c_outs.detach().cpu()
            
            _, preds = torch.max(c_outs, 1)
            # use tmp for update logger
            tmp_corrects = (preds == labels).sum()
            tmp_total = labels.size(0)
            # measure top1 accuracy.
            corrects += tmp_corrects
            total += tmp_total

            # measure information gain.
            cls_nf_mask = labels == 0
            cls_f_mask = labels == 1
            tmp_nf_ig = -1 * s_outs[cls_nf_mask] * s_outs[cls_nf_mask].log()
            tmp_f_ig = -1 * s_outs[cls_f_mask] * s_outs[cls_f_mask].log()
            nf_info_gain += tmp_nf_ig.mean()
            f_info_gain += tmp_f_ig.mean()
            
            forward_time = time.time() - end_time

            eval_logger.update(batch_id+1, 
                               data_time, 
                               forward_time, 
                               0, 
                               0,
                               0,
                               tmp_corrects/tmp_total, 
                               f_info_gain.item(),
                               nf_info_gain.item(),
                               batch_size, 
                               0)
        else:
            c_outs = model(datas)
            c_outs = c_outs.detach().cpu()
            
            _, preds = torch.max(c_outs, 1)
            # use tmp for update logger
            tmp_corrects = (preds == labels).sum()
            tmp_total = labels.size(0)
            # measure top1 accuracy.
            corrects += tmp_corrects
            total += tmp_total

            forward_time = time.time() - end_time

            eval_logger.update(batch_id+1, 
                               data_time, 
                               forward_time, 
                               0, 
                               0,
                               0,
                               tmp_corrects/tmp_total, 
                               0,
                               0,
                               batch_size, 
                               0)

        end_time = time.time()

    eval_logger.show()
    eval_acc = corrects/total
    eval_logger.log("\n\n Evaluation Accuracy:{} after {} epochs unsupervised pretrain.\n\n".format(eval_acc, epoch+1))
    return eval_acc


def main():

    args = get_args()

    train_logger, eval_logger = build_logger(args)
    
    train_logger.log("\ncreating model...")

    model = build_model(args)
    model.cuda(1)

    train_logger.log("finish.")
    train_logger.log("\nbuilding dataloader...")

    train_loader, eval_loader = build_dataloader(args)
    
    train_logger.log("finish.")
    train_logger.log("\nbuilding criterion and optimizer...")

    optimizer, schedule = build_optimizer(model, len(train_loader), args)
    criterion, cross_entropy = build_criterion(args)

    train_logger.log("finish.")
    train_logger.log("\nload pretrained: {}".format(args.load_pretrained))

    if args.load_pretrained:
        start_epochs = loadTorchModel(args.pt_path, model, optimizer)
    else:
        start_epochs = 0 # or load from pretrain model.

    # === set data number ===
    train_logger.log("\ntraining data: {:.0f} ({:.0f})".format(len(train_loader), len(train_loader.dataset)))
    eval_logger.log("\nfinetune data: {:.0f} ({:.0f})".format(len(eval_loader), len(eval_loader.dataset)))

    train_logger.set_batchs_number(len(train_loader))
    eval_logger.set_batchs_number(len(eval_loader))

    cudnn.benchmark = True
    for epoch in range(start_epochs, args.max_epochs):
        train(epoch, model, train_loader, optimizer, schedule, criterion, cross_entropy, train_logger, args)
        if (epoch + 1) % args.eval_freq == 0:
            eval(epoch, model, eval_loader, eval_logger, args)
            saveTorchModel(save_root = args.save_model_root, 
                           epoch = epoch, 
                           model = model, 
                           optimizer = optimizer)

if __name__ == "__main__":
    main()