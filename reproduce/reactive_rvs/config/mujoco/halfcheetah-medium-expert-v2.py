from reproduce.reactive_rvs.config.base import *

task = "halfcheetah-medium-expert-v2"

pretrain_critic_epoch = 2000
pretrain_command_epoch = 5000
max_epoch = 1000
eval_interval = 10
train_step_per_epoch = 1000