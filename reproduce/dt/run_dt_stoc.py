import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from acsl.buffer.toy_buffer import ToyBuffer
from acsl.env.stochastic import get_stochastic_dataset
from acsl.module.net.attention.decision_transformer import DecisionTransformer
from acsl.policy.model_free.dt import DecisionTransformerPolicy
from acsl.utils.eval import eval_decision_transformer


args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/dt/stoc_toy/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "mode": "offline", "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)
# product_root = f"./out/dt/{args.name}/{args.task}/seed{args.seed}/"

env, dataset = get_stochastic_dataset(args.task)
env.get_normalized_score = lambda x: x
obs_shape, action_shape = np.prod(env.observation_space.shape), env.action_space.n

dt = DecisionTransformer(
    obs_dim=obs_shape, action_dim=action_shape, embed_dim=args.embed_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
    seq_len=args.seq_len, episode_len=args.episode_len, 
    embed_dropout=args.embed_dropout, attention_dropout=args.attention_dropout,
    residual_dropout=args.residual_dropout
).to(args.device)

policy = DecisionTransformerPolicy(
    dt=dt, 
    state_dim=obs_shape, action_dim=action_shape, embed_dim=args.embed_dim, 
    seq_len=args.seq_len, episode_len=args.episode_len,
    action_type=args.action_type, 
    device=args.device
).to(args.device)
policy.configure_optimizers(**args.optimizer_args)
policy.train()


offline_buffer = ToyBuffer(
    dataset=dataset, 
    seq_len=args.seq_len, 
    discount=1.0, 
    return_scale=args.return_scale, 
    device=args.device
)
offline_buffer_iter = iter(DataLoader(offline_buffer, batch_size=args.train_batch_size, pin_memory=args.pin_memory, num_workers=args.num_workers))

for i_epoch in trange(1, args.max_epoch+1, desc="main loop"):
    for i_step in range(args.step_per_epoch):
        batch = next(offline_buffer_iter)
        train_metrics = policy.update(batch, args.clip_grad)
    if i_epoch % 10 == 0:
        # logger.info(f"Epoch {i_epoch}: \n{train_metrics}")
        logger.log_scalars("main_loop", train_metrics, step=i_epoch)

    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_decision_transformer(
            env,
            policy,
            target_returns=args.target_returns,
            return_scale=args.return_scale,
            n_episode=args.eval_episode,
            seed=args.seed
        )
        logger.log_scalars("eval", eval_metrics, step=i_epoch)