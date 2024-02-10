import os
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from offlinerllib.module.critic import Critic, DoubleCritic

from acsl.module.net.attention.rvs_transformer import RvSTransformer
from acsl.policy.model_free.sequence_rvs import SequenceRvS
from acsl.buffer.toy_buffer import ToyBuffer
from acsl.policy.model_free.command import InSampleMaxCommand, ConstantCommand
from acsl.utils.eval import eval_sequence_rvs, dump_sequence_rvs_actions
from acsl.env.cliff_walking import get_cliff_dataset


args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_dir=f"./log/sequence_rvs/stoc_toy/{args.name}", name=exp_name, logger_config={
    "TensorboardLogger": {}, 
    "WandbLogger": {"config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
}, activate=not args.debug)
setup(args, logger)

env, dataset = get_cliff_dataset()
env.get_normalized_score = lambda x: x
obs_shape, action_shape = np.prod(env.observation_space.shape), env.action_space.n


# ============= define critic ============== #
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
    reduce=args.v_reduce
).to(args.device)

if args.backend == "encoder-decoder":
    actor_transformer = RvSTransformer(
        obs_dim=obs_shape, action_dim=action_shape, embed_dim=args.embed_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
        seq_len=args.seq_len, episode_len=args.episode_len, 
        embed_dropout=args.embed_dropout, attention_dropout=args.attention_dropout, 
        pos_encoding=args.pos_encoding, 
        use_abs_timestep=args.use_abs_timestep, 
        decoder_type=args.decoder_type
    ).to(args.device)
elif args.backend == "gpt":
    assert False

if args.command_type == "in_sample_max":
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
    command.configure_optimizers(args.command_lr, command_weight_decay=args.command_weight_decay)
elif args.command_type == "constant":
    command = ConstantCommand(
        init=0, 
        polyak=0.995, 
        device=args.device, 
    ).to(args.device)

policy = SequenceRvS(
    critic_q=critic_q,
    critic_v=critic_v,  
    command=command, 
    actor_transformer=actor_transformer, 
    state_dim=obs_shape, action_dim=action_shape, embed_dim=args.embed_dim, 
    seq_len=args.seq_len, episode_len=args.episode_len,
    action_type=args.action_type, discount=args.discount, lambda_=args.lambda_, tau=args.tau, 
    iql_tau=args.iql_tau, 
    device=args.device
).to(args.device)
policy.train()

offline_buffer = ToyBuffer(
    dataset=dataset, 
    seq_len=args.seq_len, 
    discount=args.discount, 
    lambda_=args.lambda_, 
    return_scale=args.return_scale, 
    adv_scale=args.adv_scale, 
    use_mean_reduce=args.use_mean_reduce, 
    device=args.device)


policy.configure_optimizers(**args.optimizer_args)
offline_buffer_iter = iter(DataLoader(offline_buffer, batch_size=args.pretrain_batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers))
for i_epoch in trange(1, args.pretrain_critic_epoch+1, desc="pretrain critic"):
    for i_step in range(args.step_per_epoch):
        batch = next(offline_buffer_iter)
        train_metrics = policy.update_critic(batch)
    if i_epoch % 10 == 0:
        # logger.info(f"Epoch {i_epoch}: \n{train_metrics}")
        logger.log_scalars("pretrain", train_metrics, step=i_epoch)
del offline_buffer_iter

# illustrate the optimal action distribution
optimal_actions = np.zeros([3, 5])
for x in range(5):
    for y in range(3):
        obs = torch.FloatTensor([x, y]).repeat(action_shape).to(args.device).reshape(action_shape, obs_shape)
        actions = torch.eye(action_shape).to(args.device)
        qs = critic_q(obs, actions, reduce=False).mean(0).squeeze().detach().cpu().numpy()
        optimal_action = np.argmax(qs)
        optimal_actions[2-y, x] = optimal_action
print(optimal_actions)
        
offline_buffer.relabel(critic_q, critic_v)
offline_buffer_iter = iter(DataLoader(offline_buffer, batch_size=args.pretrain_command_batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers))
for i_epoch in trange(1, args.pretrain_command_epoch+1, desc="pretrain_command"):
    for i_step in range(args.step_per_epoch):
        batch = next(offline_buffer_iter)
        train_metrics = policy.update_command(batch)
    if i_epoch % 10 == 0:
        logger.loggers["TensorboardLogger"].log_scalars("pretrain", train_metrics, step=i_epoch)
del offline_buffer_iter

policy.configure_optimizers(**args.optimizer_args)
offline_buffer_iter = iter(DataLoader(offline_buffer, batch_size=args.train_batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers))
for i_epoch in trange(1, args.max_epoch+1, desc="main loop"):
    for i_step in range(args.step_per_epoch):
        batch = next(offline_buffer_iter)
        train_metrics = policy.update_actor(batch, args.clip_grad)
    if i_epoch % 10 == 0:
        # logger.info(f"Epoch {i_epoch}: \n{train_metrics}")
        logger.log_scalars("main_loop", train_metrics, step=i_epoch)

    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_sequence_rvs(env, policy, n_episode=args.eval_episode, seed=args.seed)
        logger.log_scalars("eval", eval_metrics, step=i_epoch)
        

# show the maximum return to go
obss, actions = dump_sequence_rvs_actions(env, policy, 10, args.seed)
for i in range(10):
    print(f"======== Test Episode {i} ==========")
    print(obss[i], actions[i])

