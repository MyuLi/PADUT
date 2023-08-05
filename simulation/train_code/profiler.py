from utils import seed_everything
seed_everything(
    seed = 3407,
    deterministic = True, 
)
from architecture import *
from utils import *
import torch
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option import opt
from tqdm import tqdm

from thop import profile, clever_format
from ptflops import get_model_complexity_info

print(opt)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

Phi_batch_test = init_mask(
    opt.mask_path, 
    opt.input_mask, 
    1, 
    device=device)

test_data = LoadTest(opt.test_path)


model = model_generator(opt, device)
    
test_gt = test_data.to(torch.float32).to(device)
input_meas = init_meas(test_gt, Phi_batch_test, opt.input_setting)

# out = model(input_meas, Phi_batch_test)


# macs, params = profile(model, inputs=([input_meas, Phi_batch_test]))
# macs, params = clever_format([macs, params], "%.3f")

# print(macs)
# print(params)

def prepare_input(resolution):
    x = torch.FloatTensor(1, 256, 310).to(device)
    Phi = torch.rand((1, 28, 256, 310)).to(device)
    return (x, Phi)


print(input_meas.shape[1:])
macs, params = get_model_complexity_info(
    model, 
    tuple(input_meas.shape[1:]),
    input_constructor=prepare_input, 
    as_strings=True,
    print_per_layer_stat=True, 
    verbose=True
)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_parameters = sum(p.numel() for p in model.parameters())

print("Requires_grad params: ", n_parameters)
print("All params: ", all_parameters)