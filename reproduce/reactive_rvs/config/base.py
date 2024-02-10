from UtilsRL.misc import NameSpace

seed = 0

task = "hopper-medium-replay-v2"

clip_grad = None
episode_len = 1000
seq_len = 16

normalize_obs = True
normalize_reward = True

embed_dim = 128

num_workers = 4
pin_memory = False

name = "debug"
class wandb(NameSpace):
    entity = "gaochenxiao"
    project = "ACSL-Onestep"

debug = False

pretrain_batch_size = 256
train_batch_size = 1024

discount = 0.99
tau = 5e-3
lambda_ = 0.00
return_scale = 1.0
adv_scale = 2.0

class optimizer_args(NameSpace):
    critic_lr = 3e-4
    actor_lr = 1e-3
    actor_weight_decay = 1e-3

pretrain_critic_epoch = 1500
pretrain_command_epoch = 1000
max_epoch = 2000
pretrain_step_per_epoch = 100
train_step_per_epoch = 100
eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50

action_type = "deterministic"

scale = 1.0

hidden_dims = [1024,1024]

expectile = 0.98
command_lr = 3e-4

v_norm_layer = False
command_norm_layer = False
actor_norm_layer = False
enhance = False

expanded_dim = 5
dropout = 0
