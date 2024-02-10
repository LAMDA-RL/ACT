from reproduce.dt.config.onestep.base import *

task = "2048-v0"
target_returns = [1.0]

normalize_obs = False
normalize_reward = False
num_layers = 2
name = "stoc"

return_scale = 1.0

seq_len = 20
episode_len = 500

max_epoch = 1500
eval_interval = 10
action_type = "categorical"