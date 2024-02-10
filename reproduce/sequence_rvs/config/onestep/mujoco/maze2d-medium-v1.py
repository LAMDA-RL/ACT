from reproduce.sequence_rvs.config.onestep.mujoco.base import *

task = "maze2d-medium-v1"
normalize_reward = True

pretrain_critic_epoch = 2000
pretrain_command_epoch = 5000
max_epoch = 5000