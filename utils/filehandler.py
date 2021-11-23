import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import json
import os


def loadJson(json_path):
    with open(json_path, "r") as fjson:
        return json.load(fjson)

def saveJson(save_path, datas):
    with open(save_path, "w") as fjson:
        fjson.write(json.dumps(datas, indent=2))

def loadTxt(txt_path):
    with open(txt_path, "r") as ftxt:
        return ftxt.read()

def saveTxt(save_path: str, msg: str, keep_old: bool = True): 
    save_msg = ""
    if keep_old:
        if os.path.isfile(save_path):
            save_msg += loadTxt(save_path)

    save_msg += msg + "\n"
    with open(save_path, "w") as ftxt:
        ftxt.write(save_msg)

def saveTorchModel(save_root: str, 
                   epoch: int, 
                   model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.SGD] = None, 
                   save_name: Optional[str] = None):
    
    if save_name is None:
        save_name = "e{}.pt".format(epoch)
    assert save_name.split(".")[-1] == "pt"

    if optimizer is None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict()
            }, save_root + save_name)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, save_root + save_name)


def loadTorchModel(pt_path: str,
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.SGD] = None):
    checkpoint = torch.load(pt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

    