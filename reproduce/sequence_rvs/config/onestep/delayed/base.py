from reproduce.sequence_rvs.config.onestep.base import *

command_weight_decay = 0.0
delayed_reward = True
adv_scale = 10.0
eval_interval = 100
lambda_ = 0.7
v_reduce = "mean"
pretrain_critic_epoch = 3000

tau = 0.05