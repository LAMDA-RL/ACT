import os
import copy
import torch
import wandb
import numpy as np
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from torch.utils.data import DataLoader

# from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.env.d4rl import get_d4rl_dataset
from offlinerllib.module.critic import DoubleCritic

from acsl.buffer.act_buffer import ACTTrajectoryBuffer
from acsl.policy.model_free.iql import IQL


args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_dir=f"./log_reproduce/{args.name}", name=exp_name, logger_config={
    "TensorboardLogger": {}, 
    "WandbLogger": {"config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
}, activate=not args.debug, backup_stdout=True)
setup(args, logger)
# setup(args)
product_root = f"./{args.save_path}/{args.name}/{args.task}/"
if not os.path.exists(product_root):
    os.makedirs(product_root)

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward, discard_last=False)
obs_shape, action_shape = np.prod(env.observation_space.shape), env.action_space.shape[-1]

critic_q = DoubleCritic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape + action_shape, 
    hidden_dims=args.hidden_dims, 
    norm_layer=args.q_norm_layer, 
    reduce="min", 
).to(args.device)

critic_v = DoubleCritic(
    backend=torch.nn.Identity(), 
    input_dim=obs_shape, 
    hidden_dims=args.hidden_dims, 
    norm_layer=args.v_norm_layer, 
    reduce=args.v_reduce, 
).to(args.device)

iql = IQL(
    critic_q=critic_q, 
    critic_v=critic_v, 
    v_target=args.v_target, 
    expectile=args.iql_tau, 
    tau=args.tau, 
    discount=args.discount, 
    device=args.device
).to(args.device)
iql.configure_optimizers(args.optimizer_args.critic_lr)

offline_buffer = ACTTrajectoryBuffer(
    dataset=dataset, 
    seq_len=args.seq_len, 
    discount=args.discount, 
    lambda_=args.lambda_, 
    return_scale=args.return_scale, 
    adv_scale=args.adv_scale, 
    use_mean_reduce=args.use_mean_reduce, 
    delayed_reward=args.delayed_reward, 
    device=args.device)

best_ckpts = (None, None)
best_loss = 9e9
if args.load_path is None:
    logger.info("Start training ...")
    offline_buffer_iter = iter(DataLoader(offline_buffer, batch_size=args.pretrain_batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers))
    for i_epoch in trange(1, args.pretrain_critic_epoch+1, desc="pretrain critic"):
        for i_step in range(args.step_per_epoch):
            batch = next(offline_buffer_iter)
            train_metrics = iql.update(batch)
        if i_epoch % 10 == 0:
            logger.log_scalars("pretrain", train_metrics, step=i_epoch)
        if i_epoch >= int(0.7*args.pretrain_critic_epoch) and i_epoch % args.valid_interval == 0:
            cur_loss = iql.evaluate(offline_buffer)
            if cur_loss < best_loss:
                best_ckpts = copy.deepcopy(critic_q.state_dict()), copy.deepcopy(critic_v.state_dict())
                best_loss = cur_loss
                logger.info(f"Updating best checkpoint at epoch {i_epoch}, loss is {best_loss}")
    critic_q.load_state_dict(best_ckpts[0])
    critic_v.load_state_dict(best_ckpts[1])
    iql.save(product_root)
else:
    iql.load(args.load_path)

# mse_v_loss, mse_q_loss, exp_q_loss = iql.evaluate(offline_buffer)
# with open(os.path.join(product_root, "statistics.txt"), "w") as fp:
#     fp.write(f"expectile {args.iql_tau}\n")
#     fp.write(f"mse_v_loss {mse_v_loss}\n")
#     fp.write(f"mse_q_loss {mse_q_loss}\n")
#     fp.write(f"exp_q_loss {exp_q_loss}\n")
    
# if args.load_path is None:
#     iql.save(product_root)