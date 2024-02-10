from reproduce.reactive_rvs.config.base import *
import acsl.env.stochastic_mujoco

task = "walker2d-stochastic-0.35-medium-v3"

pretrain_critic_epoch = 2000
pretrain_command_epoch = 3000
max_epoch = 1500
train_step_per_epoch = 1000