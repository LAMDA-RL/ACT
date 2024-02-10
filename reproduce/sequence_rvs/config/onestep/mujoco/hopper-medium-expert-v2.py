from reproduce.sequence_rvs.config.onestep.mujoco.base import *

task = "hopper-medium-expert-v2"

pretrain_critic_epoch = 2000
pretrain_command_epoch = 5000
max_epoch = 10000

num_layers = 3
num_heads = 1
optimizer_args.actor_weight_decay = 1e-4