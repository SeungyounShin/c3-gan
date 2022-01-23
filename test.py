import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from datasets import Dataset
from utils import *
from evals import *
from inception import InceptionV3

from train import *

num_gt_cls = cfg.NUM_GT_CLASSES
over = cfg.OVER
num_cls = num_gt_cls*over
bs = 1

netG, netD = load_network(num_cls, cfg.DEVICE)

model_dict = torch.load("./output/voc/best_fid.pt")
gen_weight     = model_dict['netG']
dis_weight = model_dict['netD']

netG.load_state_dict(gen_weight)
netD.load_state_dict(dis_weight)
netG.eval()
netD.eval()

print(" + Model loaded")

rand_z = torch.FloatTensor(bs, cfg.GAN.Z_DIM).normal_(0, 1).to(cfg.DEVICE)
rand_cz = torch.FloatTensor(bs, cfg.GAN.CZ_DIM).normal_(0, 1).to(cfg.DEVICE)
rand_c = torch.zeros(bs, num_cls).to(cfg.DEVICE)
rand_idx = [i for i in range(num_cls)]
random.shuffle(rand_idx)
for i, idx in enumerate(rand_idx[:bs]):
    rand_c[i, idx] = 1

bg_img, fg_mask, fg_img, fake_img = netG(rand_z, rand_cz, rand_c)
self.bg_img = postprocess(self.bg_img)
self.fg_mask = postprocess(self.fg_mask.repeat(1,3,1,1))
self.fg_img = postprocess(self.fg_img)
self.fake_img = postprocess(self.fake_img)
