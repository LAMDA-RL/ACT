from reproduce.sequence_rvs.config.onestep.base import *

discount = 1.0
task = "2048-v0"
episode_len = 30
seq_len = 5
pretrain_critic_epoch = 2500
pretrain_command_epoch = 2500
max_epoch = 1000

num_layers = 2   # shrink the layer num

lambda_ = None
action_type = "categorical"
hidden_dims = [256, 256]

eval_interval = 50

pretrain_batch_size = 64
pretrain_command_batch_size = 64
v_reduce = "mean"

name = "stoc_toy"