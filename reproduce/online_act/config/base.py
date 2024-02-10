from UtilsRL.misc import NameSpace

seed = 0

task = None

clip_grad = 0.25
episode_len = 1000
seq_len = 20
episode_len = 1000

normalize_obs = False
normalize_reward = False

embed_dim = 512
embed_dropout = 0.1
attention_dropout = 0.1

num_heads = 4
num_layers = 3
num_workers = 4
pin_memory = True

name = "debug"
class wandb(NameSpace):
    entity = "gaochenxiao"
    project = "ACSL-Online"

debug = False

discount = 0.99
tau = 5e-3
lambda_ = 0.0
return_scale = 1.0
adv_scale = None

class optimizer_args(NameSpace):
    critic_lr = 3e-4
    actor_lr = 1e-4
    critic_weight_decay = 1e-4
    actor_weight_decay = 1e-3
    critic_betas = [0.9, 0.999]
    actor_betas = [0.9, 0.999]
    critic_lr_scheduler_fn = lambda step: min((step+1)/10000, 1)
    actor_lr_scheduler_fn = lambda step: min((step+1)/10000, 1)


num_rollouts = 1 
pretrain_epoch = 5
online_epoch = 1000
batch_size = 256
critic_step_per_epoch = 1000
act_step_per_epoch = 300

eval_episode = 10
eval_interval = 10
log_interval = 10
save_interval = 50

action_type = "stochastic"

scale = 1.0

hidden_dims = [256, 256, 256]

expectile = 0.98
command_lr = 3e-4

v_norm_layer = False
command_norm_layer = False
enhance = False
pos_encoding = "embedding"

buffer_size = 2000
load_path = None
add_noise = True
noise_scale = 0.1
