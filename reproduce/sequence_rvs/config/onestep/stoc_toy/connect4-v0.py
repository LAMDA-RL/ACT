from reproduce.sequence_rvs.config.onestep.base import *

discount = 1.0
task = "connect4-v0"
episode_len = 50
seq_len = 20


pretrain_critic_epoch = 2000
pretrain_command_epoch = 2000
max_epoch = 2000

num_layers = 2   # shrink the layer num

lambda_ = None
action_type = "categorical"
hidden_dims = [256, 256]

eval_interval = 200

pretrain_batch_size = 64
pretrain_command_batch_size = 64
v_reduce = "mean"