import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.policy.model_free.dt import DecisionTransformerPolicy
from offlinerllib.module.net.attention.dt import DecisionTransformer
# from offlinerllib.utils.eval import eval_decision_transformer

# from offlinerllib.buffer import D4RLTrajectoryBuffer
from acsl.buffer.act_buffer import ACTTrajectoryBuffer
from acsl.utils.eval import eval_decision_transformer


args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/dt/onestep/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "mode": "offline", "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)
product_root = f"./out/dt/{args.name}/{args.task}/seed{args.seed}/"

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward, discard_last=False)
obs_shape, action_shape = np.prod(env.observation_space.shape), env.action_space.shape[-1]


dt = DecisionTransformer(
    obs_dim=obs_shape, action_dim=action_shape, embed_dim=args.embed_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
    seq_len=args.seq_len, episode_len=args.episode_len, 
    embed_dropout=args.embed_dropout, attention_dropout=args.attention_dropout,
    residual_dropout=args.residual_dropout
).to(args.device)


policy = DecisionTransformerPolicy(
    dt=dt, 
    state_dim=obs_shape, action_dim=action_shape,
    seq_len=args.seq_len, episode_len=args.episode_len,
    device=args.device
).to(args.device)
policy.configure_optimizers(**args.optimizer_args)
policy.train()

# ================ fit dt =========================== #
offline_buffer = ACTTrajectoryBuffer(
    dataset=dataset, 
    seq_len=args.seq_len, 
    discount=1.0, # DT don't discount rewards
    return_scale=args.return_scale, 
    delayed_reward=args.delayed_reward, 
    device=args.device
)
offline_buffer_iter = iter(DataLoader(offline_buffer, batch_size=args.train_batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers))

for i_epoch in trange(1, args.max_epoch+1, desc="main loop"):
    for i_step in range(args.step_per_epoch):
        batch = next(offline_buffer_iter)
        train_metrics = policy.update(batch, args.clip_grad)
    if i_epoch % 10 == 0:
        logger.log_scalars("main_loop", train_metrics, step=i_epoch)

    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_decision_transformer(
            env,
            policy,
            target_returns=args.target_returns,
            return_scale=args.return_scale,
            delayed_reward=args.delayed_reward, 
            n_episode=args.eval_episode,
            seed=args.seed
        )
        logger.log_scalars("eval", eval_metrics, step=i_epoch)
        logger.info(f"Epoch {i_epoch}: \n{eval_metrics}")
        
