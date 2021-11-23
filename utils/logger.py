import os
import time
import json
from utils.filehandler import loadJson, saveJson, loadTxt, saveTxt

class AverageMeter(object):
    """
    computes and stores the average and current value
    """
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


class MyLogger(object):

    def __init__(self, save_path, prefix):
        self.save_path = save_path
        with open(save_path, "w") as ftxt:
            ftxt.write("Logger:" + prefix)
            ftxt.write("\n")
        self.epoch = -1
        self.batch_id = -1
        self.lr = -1
        self.data_times = AverageMeter()
        self.forward_times = AverageMeter()
        self.losses = AverageMeter()
        self.ig_losses = AverageMeter()
        self.scores_losses = AverageMeter()
        self.accs = AverageMeter()
        self.foul_ig = AverageMeter()
        self.nofoul_ig = AverageMeter()
        self.total_batchs = 0 # in one epoch.

    def set_batchs_number(self, batchs):
        self.total_batchs = batchs
        
    def update(self, batch_id, data_time, forward_time, loss, scores_loss, infogain_loss, acc, foul_infogain, nofoul_infogain, batch_size, lr):
        self.data_times.update(data_time)
        self.forward_times.update(forward_time)
        self.losses.update(loss, batch_size)
        self.ig_losses.update(infogain_loss, batch_size)
        self.scores_losses.update(scores_loss, batch_size)
        self.accs.update(acc, batch_size)
        self.foul_ig.update(foul_infogain, batch_size)
        self.nofoul_ig.update(nofoul_infogain, batch_size)
        self.lr = lr
        self.batch_id = batch_id

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.data_times.reset()
        self.forward_times.reset()
        self.losses.reset()
        self.ig_losses.reset()
        self.accs.reset()
        self.foul_ig.reset()
        self.nofoul_ig.reset()
        self.scores_losses.reset()

    def log(self, msg):
        print(msg)
        saveTxt(save_path = self.save_path, 
                msg = msg)

    def res_time(self):
        avg_t = self.forward_times.avg
        res_batch = self.total_batchs - self.batch_id
        res_t = res_batch * avg_t
        hours = int(res_t/3600)
        mins = int((res_t - 3600*hours) / 60)
        secs = int(res_t - 3600*hours - 60*mins)
        return [hours, mins, secs]

    def show(self):
        msg = "\n"\
              "Epoch:{0}-{1}\n"\
              "forward: {forward_time.val:.3f} ({forward_time.avg:.3f}) secs/batch, data: {data_time.val:.3f} ({data_time.avg:.3f}) secs/batch\n"\
              "loss: {loss.val:.4f} ({loss.avg:.4f}), scores-loss: {scores_losses.val:.4f} ({scores_losses.avg:.4f}), infogain-loss: {ig_losses.val:.4f} ({ig_losses.avg:.4f}), acc: {acc.val:.2f} ({acc.avg:.2f})\n"\
              "nofouls-infogain: {nofoul_ig.val:.4f} ({nofoul_ig.avg:.4f}), fouls-infogain: {foul_ig.val:.4f} ({foul_ig.avg:.4f}), lr: {lr:.4f}\n"\
              "remaining: {res_time[0]:.0f} hours {res_time[1]:.0f} mins, {res_time[2]:.0f} secs"\
              .format(
                                self.epoch, 
                                self.batch_id,  
                                forward_time=self.forward_times, 
                                data_time=self.data_times, 
                                loss=self.losses, 
                                ig_losses=self.ig_losses,
                                scores_losses=self.scores_losses,
                                acc=self.accs, 
                                nofoul_ig = self.nofoul_ig,
                                foul_ig = self.foul_ig,
                                lr=self.lr, 
                                res_time=self.res_time()
                              )
        print(msg)
        saveTxt(save_path = self.save_path, 
                msg = msg)
