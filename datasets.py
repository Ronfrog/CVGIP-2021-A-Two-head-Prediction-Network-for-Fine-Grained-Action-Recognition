import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.io as io
import cv2
import random
import time
import numpy as np
import os
import json

from utils.filehandler import loadJson


class BasketballFoulsDataset(nn.Module):

    def __init__(self, 
                 istrain: bool = True,
                 data_paths_json: str = "./clips/train1.json",
                 videos_root: str = "./clips/", 
                 data_size: tuple = (64, 3, 224, 224),
                 select_type: str = "frequency",
                 select_freq: int = 1,
                 select_random_start: bool = True,
                 select_random_interval: int = 8,
                 ):
        """
        select_type ["frequency", "equally",]
        """
        random.seed(time.time())
        np.random.seed(int(time.time()))
        self.istrain = istrain
        self.data_paths_json = data_paths_json
        self.videos_root = videos_root
        self.video_extensions = ["mp4", "avi"]
        self.data_size = data_size
        self.select_type = select_type
        self.select_freq = select_freq
        self.select_random_interval = select_random_interval
        self.select_random_start = select_random_start
        self.data_infos, self.foul_paths, self.nofoul_paths = self.getDataInfos()

        self.weak_trans =  transforms.Compose([
                    transforms.Resize((data_size[2], data_size[3])),
                    transforms.RandomCrop(data_size[2], padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])


    def __len__(self):
        return len(self.data_infos)

    def getDataInfos(self) -> [list, list, list]:

        choosen_paths = loadJson(self.data_paths_json)

        data_infos = []
        foul_paths = []
        nofoul_paths = []
        
        for data in choosen_paths:
            
            video_abspath = self.videos_root + data[0]
            label = data[1]

            data_infos.append({"path":video_abspath, "label":label})
            if label == 0:
                nofoul_paths.append(video_abspath)
            else:
                foul_paths.append(video_abspath)

        return data_infos, foul_paths, nofoul_paths

    def select_frames(self, vlength: int) -> torch.Tensor:
        
        start_id = 0
        if self.select_random_start:
            start_id = random.randint(0, self.select_random_interval-1)

        if self.select_type == "frequency":
            select_ids = torch.arange(self.data_size[0]) + start_id
            select_ids *= self.select_freq
            select_ids = torch.remainder(select_ids, vlength)

        elif self.select_type == "equally":
            if vlength >= self.data_size[0]:
                avg_interval = vlength/self.data_size[0]
                select_ids = torch.arange(0, vlength, avg_interval)
                select_ids = select_ids.round()
            else:
                # equal to frequency == 1 situation
                select_ids = torch.arange(self.data_size[0]) + start_id
                select_ids = torch.remainder(select_ids, vlength)

        elif self.select_type == "random":
            if vlength >= self.data_size[0]:
                select_ids = np.random.choice(vlength, self.data_size[0], replace=False) + start_id
                select_ids = torch.from_numpy(select_ids.sort())
            else:
                # equal to frequency == 1 situation
                select_ids = torch.arange(self.data_size[0]) + start_id
                select_ids = torch.remainder(select_ids, vlength)
        else:
            raise ValueError("select_type only support 'frequency', 'equally', 'random'.")

        return select_ids.long()



    def __getitem__(self, index):
        vpath = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]

        # select frame and transforms.
        vframes, _, _ = io.read_video(vpath, pts_unit='sec')
        vframes = vframes.permute(0, 3, 1, 2).float() # T, H, W, C --> T, C, H, W
        select_ids = self.select_frames(vframes.size(0))
        vframes = self.weak_trans(vframes[select_ids])

        if self.istrain:
            pair_index, pair_label = None, None
            if label == 0:
                pair_index = random.randint(0, len(self.foul_paths)-1)
                pair_vpath = self.foul_paths[pair_index]
                pair_label = 1
            else:
                pair_index = random.randint(0, len(self.nofoul_paths)-1)
                pair_vpath = self.nofoul_paths[pair_index]
                pair_label = 0

            pair_vframes, _, _ = io.read_video(pair_vpath, pts_unit='sec')
            # select frame and transforms.
            pair_vframes = pair_vframes.permute(0, 3, 1, 2).float() # T, H, W, C --> T, C, H, W
            pair_select_ids = self.select_frames(pair_vframes.size(0))
            pair_vframes = self.weak_trans(pair_vframes[pair_select_ids])

            return vframes.permute(1, 0, 2, 3), label, pair_vframes.permute(1, 0, 2, 3), pair_label

        return vframes.permute(1, 0, 2, 3), label

