import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from offlinerllib.buffer import D4RLTransitionBuffer, D4RLTrajectoryBuffer
from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.module.critic import Critic, DoubleCritic
from offlinerllib.module.net.mlp import MLP

from acsl.policy.model_free.act_gpt import ACTGPTPolicy
from acsl.module.net.attention.act import AdvantageConditionedTransformer
from acsl.buffer.act_buffer import ACTTrajectoryBuffer
from acsl.policy.model_free.command import InSampleMaxCommand
from acsl.utils.eval import eval_act_gpt

"""
TODO: 
  - check whether relabel is needed   
  - check whether positional encoding matters
  - check whether masking previous command tokens matters
  - check whether to add relu layer after input positional encoding
  - check whether to add dropout for positional encoding
"""


args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/act_gpt/onestep/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)
product_root = f"./out/act_gpt/onestep/{args.name}/{args.task}/seed{args.seed}/"

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward, discard_last=False)
obs_shape, action_shape = env.observation_space.shape[0], env.action_space.shape[-1]

# ========== define Q network, V network, command_agent and command_model =============
critic_q = DoubleCritic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape + action_shape, 
    hidden_dims=args.hidden_dims, 
    reduce="min", 
).to(args.device)

critic_v = DoubleCritic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape, 
    hidden_dims=args.hidden_dims, 
    norm_layer=args.v_norm_layer, 
    reduce="min", 
).to(args.device)

act = AdvantageConditionedTransformer(
    obs_dim=obs_shape, 
    action_dim=action_shape, 
    embed_dim=args.embed_dim, 
    num_layers=args.num_layers, 
    seq_len=args.seq_len, 
    episode_len=args.episode_len, 
    num_heads=args.num_heads, 
    attention_dropout=args.attention_dropout, 
    residual_dropout=args.residual_dropout, 
    embed_dropout=args.embed_dropout
).to(args.device)
command_backend = Critic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape, 
    hidden_dims=args.hidden_dims, 
    norm_layer=args.command_norm_layer
).to(args.device)
command = InSampleMaxCommand(
    command_module=command_backend, 
    is_agent=True, 
    expectile=args.expectile, 
    enhance=args.enhance, 
    device=args.device
).to(args.device)
command.configure_optimizers(args.command_lr)

policy = ACTGPTPolicy(
    act=act, 
    critic_q=critic_q, 
    critic_v=critic_v, 
    command=command, 
    state_dim=obs_shape, 
    action_dim=action_shape, 
    embed_dim=args.embed_dim, 
    seq_len=args.seq_len, 
    episode_len=args.episode_len, 
    action_type=args.action_type, 
    discount=args.discount, 
    lambda_=args.lambda_, 
    tau=args.tau, 
    device=args.device
).to(args.device)
policy.configure_optimizers(**args.optimizer_args)
policy.train()


# ================ pretrain critic and command agent ================= #
offline_buffer = ACTTrajectoryBuffer(
    dataset=dataset, 
    seq_len=args.seq_len, 
    discount=args.discount, 
    lambda_=args.lambda_, 
    return_scale=args.return_scale, 
    adv_scale=args.adv_scale, 
    device=args.device)
offline_buffer_iter = iter(DataLoader(offline_buffer, batch_size=args.pretrain_batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers))
for i_epoch in trange(1, args.pretrain_critic_epoch+1, desc="pretrain critic"):
    for i_step in range(args.step_per_epoch):
        batch = next(offline_buffer_iter)
        train_metrics = policy.update_critic(batch)
    if i_epoch % 10 == 0:
        # logger.info(f"Epoch {i_epoch}: \n{train_metrics}")
        logger.loggers["TensorboardLogger"].log_scalars("pretrain", train_metrics, step=i_epoch)

# ================ relabel the dataset to advantage =========== #
del offline_buffer_iter
offline_buffer.relabel(critic_q, critic_v)
offline_buffer_iter = iter(DataLoader(offline_buffer, batch_size=args.pretrain_batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers))
        
for i_epoch in trange(1, args.pretrain_command_epoch+1, desc="pretrain_command"):
    for i_step in range(args.step_per_epoch):
        batch = next(offline_buffer_iter)
        train_metrics = policy.update_command(batch)
    if i_epoch % 10 == 0:
        logger.loggers["TensorboardLogger"].log_scalars("pretrain", train_metrics, step=i_epoch)


# ================ fit act gpt =========================== #
policy.train()
offline_buffer_iter = iter(DataLoader(offline_buffer, batch_size=args.train_batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers))
for i_epoch in trange(1, args.max_epoch+1, desc="main loop"):
    for i_step in range(args.step_per_epoch):
        batch = next(offline_buffer_iter)
        train_metrics = policy.update(batch, args.clip_grad)
    if i_epoch % 10 == 0:
        # logger.info(f"Epoch {i_epoch}: \n{train_metrics}")
        logger.log_scalars("main_loop", train_metrics, step=i_epoch)

    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_act_gpt(args.relabel, env, policy, n_episode=args.eval_episode, seed=args.seed)
        logger.log_scalars("eval", eval_metrics, step=i_epoch)