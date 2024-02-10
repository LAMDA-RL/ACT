# Advantage-Conditioned Transformer
This is the official code for [*ACT: Empowering Decision Transformer with Dynamic Programming via Advantage Conditioning*](https://arxiv.org/abs/2309.05915), which is accepted by AAAI 2024.

## Dependencies
```bash
offlinerllib==0.1.4
UtilsRL==0.6.5
gym==0.23.1
mujoco-py==2.1.2.14
torch==2.0.1
D4RL==1.1
```

## How to Reproduce the results
Below are the commands for reproducing the results. Feel free to contact me if anything goes wrong in your local dev environment. 

+ For D4RL tasks
    ```bash
    python3 reproduce/sequence_rvs/run_sequence_rvs_onestep.py \
        --config reproduce/sequence_rvs/config/onestep/mujoco/${env_name}-v2.py \
        --iql_tau ${iql_tau}
    ```
    Here the value for iql tau can be found in the Appendix. 
    In the actual benchmarking we incorporated model selection for the critics. If you want to do that, you can use `reproduce/sequence_rvs/run_iql_pretrain.py` t first pre-train the critics, select the best fitted one, and add `--load_path` to load the selected critics. 
+ For the 2048 game
    ```bash
    python3 reproduce/sequence_rvs/run_sequence_rvs_stoc.py \
        --config reproduce/sequence_rvs/config/onestep/stoc_toy/2048-v0.py 
    ```
+ For the delayed rewards tasks
    ```bash
    python3 reproduce/sequence_rvs/run_sequence_rvs_onestep.py \
        --config reproduce/sequence_rvs/config/onestep/delayed/base.py \
        --task walker2d-medium-expert-v2
    ```
+ For the stochastic mujoco tasks
    ```bash
    python3 reproduce/sequence_rvs/run_sequence_rvs_onestep.py \
        --config reproduce/sequence_rvs/config/onestep/stochastic_mujoco/
    ```

## Citation
```
@inproceedings{act,
  author = {Chen-Xiao Gao, Chenyang Wu, Mingjun Cao, Rui Kong, Zongzhang Zhang, Yang Yu}, 
  title = {ACT: Empowering Decision Transformer with Dynamic Programming via Advantage Conditioning},
  booktitle = {Proceedings of the Thirty-Eighth {AAAI} Conference on Artificial Intelligence},
  year = {2024},
}
```