# Copyright (c) 2020 Uber Technologies, Inc. All rights reserved.

from avlearn.modules.predictors.lanegcn.utils import StepLR

# Train
seed = 42
epoch = 0
opt = "adam"
lr = [1e-3, 1e-4]
lr_epochs = [32]
lr_func = StepLR(lr, lr_epochs)

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
