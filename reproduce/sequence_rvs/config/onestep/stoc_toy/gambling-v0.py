from reproduce.sequence_rvs.config.onestep.base import *

discount = 1.0
task = "gambling-v0"
episode_len = 5
seq_len = 5

lambda_ = None
action_type = "categorical"
hidden_dims = [256, 256]
command_weight_decay = 5e-4
pos_encoding = "sinusoid"