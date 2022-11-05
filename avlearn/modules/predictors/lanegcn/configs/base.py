# Copyright (c) 2020 Uber Technologies, Inc. All rights reserved.

import os
from pathlib import Path
from avlearn.modules.predictors.lanegcn.utils import StepLR

file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)

# Train
seed = 42
epoch = 0
opt = "adam"
num_epochs = 36
lr = [1e-3, 1e-4]
lr_epochs = [32]
lr_func = StepLR(lr, lr_epochs)
save_dir = Path(root_path) / Path("results")

if not os.path.isabs(save_dir):
    save_dir = os.path.join(root_path, "results", save_dir)

batch_size = 16
val_batch_size = 16
workers = 0
val_workers = workers

train_size = 19
past_window = 2.0
future_window = 20.0

# Model
rot_aug = False
pred_range = [-100.0, 100.0, -100.0, 100.0]
num_scales = 6
n_actor = 128
n_map = 128
actor2map_dist = 7.0
map2actor_dist = 6.0
actor2actor_dist = 100.0
pred_size = 12
pred_step = 1
num_preds = pred_size // pred_step
num_mods = 10
cls_coef = 1.0
reg_coef = 1.0
mgn = 0.2
cls_th = 2.0
cls_ignore = 0.2
