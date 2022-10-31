# Copyright (c) 2020 Uber Technologies, Inc. All rights reserved.

import os
from avlearn.modules.predictors.lanegcn.model.utils import StepLR

### config ###
file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]

"""Train"""
display_iters = 205942
val_iters = 205942 * 2
save_freq = 1.0
epoch = 0
horovod = False
opt = "adam"
num_epochs = 36
lr = [1e-3, 1e-4]
lr_epochs = [32]
lr_func = StepLR(lr, lr_epochs)
save_dir = os.path.join(root_path, "results", model_name)

if not os.path.isabs(save_dir):
    save_dir = os.path.join(root_path, "results", save_dir)

batch_size = 16
val_batch_size = 16
workers = 0
val_workers = workers


"""Dataset"""
# Raw Dataset
train_split = os.path.join(
    root_path, "dataset/train/data"
)
val_split = os.path.join(root_path, "dataset/val/data")
test_split = os.path.join(root_path, "dataset/test_obs/data")

# Preprocessed Dataset
preprocess = True  # whether use preprocess or not
preprocess_train = os.path.join(
    root_path, "dataset", "preprocess", "train_crs_dist6_angle90.p"
)
preprocess_val = os.path.join(
    root_path, "dataset", "preprocess", "val_crs_dist6_angle90.p"
)
preprocess_test = os.path.join(root_path, "dataset", 'preprocess',
                               'test_test.p')
train_size = 19

"""Model"""
rot_aug = False
pred_range = [-100.0, 100.0, -100.0, 100.0]
num_scales = 6
n_actor = 128
n_map = 128
actor2map_dist = 7.0
map2actor_dist = 6.0
actor2actor_dist = 100.0
pred_size = 30
pred_step = 1
num_preds = pred_size // pred_step
num_mods = 6
cls_coef = 1.0
reg_coef = 1.0
mgn = 0.2
cls_th = 2.0
cls_ignore = 0.2
### end of config ###
