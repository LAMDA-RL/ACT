from UtilsRL.misc import NameSpace
import acsl.env.stochastic_mujoco

seed = 0

task = "hopper-medium-replay-v2"

clip_grad = 0.25
episode_len = 1000
seq_len = 20

normalize_obs = True
normalize_reward = False

embed_dim = 128
embed_dropout = 0.1
attention_dropout = 0.1
residual_dropout = 0.1

num_heads = 1
num_layers = 6
num_workers = 4
pin_memory = True

name = "debug"
class wandb(NameSpace):
    entity = "gaochenxiao"
    project = "ACSL-Onestep"

debug = False

train_batch_size = 64
return_scale = 1e-3

class optimizer_args(NameSpace):
    lr = 1e-4
    weight_decay = 1e-4
    betas = [0.9, 0.999]
    warmup_steps = 10000

max_epoch = 5000
step_per_epoch = 100
eval_episode = 10
eval_interval = 40
log_interval = 10
save_interval = 50

delayed_reward = False