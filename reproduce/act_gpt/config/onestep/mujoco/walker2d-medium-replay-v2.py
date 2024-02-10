from reproduce.act_gpt.config.onestep.base import *

task = "walker2d-medium-replay-v2"

pretrain_critic_epoch = 2000
pretrain_command_epoch = 5000
max_epoch = 5000
expectile = 0.98