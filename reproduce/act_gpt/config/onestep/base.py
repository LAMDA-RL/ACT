from UtilsRL.misc import NameSpace

seed = 0

task = "hopper-medium-replay-v2"

clip_grad = 0.25
episode_len = 1000
seq_len = 20

normalize_obs = True
normalize_reward = True

embed_dim = 128
embed_dropout = 0.1
attention_dropout = 0.1
residual_dropout = 0.1

num_heads = 1
num_layers = 3
num_workers = 4
pin_memory = True

name = "debug"
class wandb(NameSpace):
    entity = "gaochenxiao"
    project = "ACSL-GPT-Onestep"

debug = False

pretrain_batch_size = 256
train_batch_size = 64

discount = 0.99
tau = 5e-3
lambda_ = 0.0
return_scale = 1.0
adv_scale = 2.0

class optimizer_args(NameSpace):
    critic_lr = 3e-4
    actor_lr = 1e-4
    critic_weight_decay = 1e-4
    actor_weight_decay = 1e-3
    critic_betas = [0.9, 0.999]
    actor_betas = [0.9, 0.999]
    critic_lr_scheduler_fn = lambda step: min((step+1)/10000, 1)
    actor_lr_scheduler_fn = lambda step: min((step+1)/10000, 1)

pretrain_critic_epoch = 1500
pretrain_command_epoch = 1000
max_epoch = 2000
step_per_epoch = 100
eval_episode = 10
eval_interval = 40
log_interval = 10
save_interval = 50

action_type = "deterministic"

scale = 1.0

hidden_dims = [256, 256, 256]

expectile = 0.98
command_lr = 3e-4

v_norm_layer = False
command_norm_layer = False
enhance = False

relabel = False