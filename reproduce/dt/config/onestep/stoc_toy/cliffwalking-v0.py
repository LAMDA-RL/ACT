from reproduce.dt.config.onestep.base import *

task = "cliffwalking-v0"
target_returns = [11.0]

normalize_obs = False
normalize_reward = False
num_layers = 2
name = "stoc"

return_scale = 1.0
max_epoch = 1000
eval_interval = 50
action_type = "categorical"

episode_len = 30
seq_len = 5
