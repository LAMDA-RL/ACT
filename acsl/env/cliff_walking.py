import gym
from gym import spaces
import numpy as np
# from gym.utils import EzPickle


class CliffWalking(gym.Env):
    def __init__(self, epsilon=0.3, penalty=-25.0, reward = 15.0):
        self.epsilon = epsilon
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(2, ), 
            dtype=np.float32
        )
        self.epsilon = epsilon
        self.penalty = penalty
        self.reward = reward
        self.board = np.zeros([5, 3])

    def reset(self):
        self._step = 0
        self.cur_pos = [0, 0]
        self.board = np.zeros([5, 3])
        self.board[self.cur_pos[0], self.cur_pos[1]] = 1
        return np.asarray(self.cur_pos)
    
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        # left: 0, right: 1, up: 2, down: 3
        self.board[self.cur_pos[0], self.cur_pos[1]] = 0
        fall = False
        
        # action of left and right will become up and down randomly 
        if action in {0, 1}:
            eps = np.random.random()
            if eps > 1 - self.epsilon / 2:
                action = 2
            elif eps > 1 - self.epsilon and eps <= 1 - self.epsilon / 2:
                action = 3
        
        # handle the real action
        if action in {2, 3}:
            if action == 2:
                next_y = self.cur_pos[1] + 1
            elif action == 3:
                next_y = self.cur_pos[1] - 1
            if next_y not in {0, 1 ,2}:
                fall = True
                next_y = np.clip(next_y, 0, 2)
            self.cur_pos[1] = next_y
        elif action in {0, 1}:
            if action == 0:
                next_x = self.cur_pos[0] - 1
            elif action == 1:
                next_x = self.cur_pos[0] + 1
            if next_x not in {0, 1, 2, 3, 4}:
                fall = True
                next_x = np.clip(next_x, 0, 4)
            self.cur_pos[0] = next_x
            
        self.board[self.cur_pos[0], self.cur_pos[1]] = 1
        self._step += 1
        
        done = False
        reward = -1
        info = {
            "success": False, 
            "fall": False
        }
        if self.cur_pos[0] == 4 and self.cur_pos[1] == 0:
            reward += self.reward
            done = True
            info["success"] = True
        if fall:
            reward += self.penalty
            done = True
            info["fall"] = True
        if self._step >= 30:
            done = True
        return np.asarray(self.cur_pos), reward, done, info
    
    def render(self):
        for x in range(3):
            print("| ", end="")
            for y in self.board[..., 2-x]:
                print(f"{y} |", end="")
            print("")
        print("--------------------------")
        
        
class BlindWalker():
    def __init__(self, row):
        self.row = row
        
    def select_action(self, obs):
        col, row = obs[0], obs[1]
        if col == 0:
            if row < self.row:
                return 2
            elif row > self.row:
                return 3
        if col == 4:
            return 3
        return 1
    
class OptimalWalker():
    def __init__(self, *args, **kwargs):
        pass
    
    def select_action(self, obs):
        col, row = obs[0], obs[1]
        if col == 4 :
            return 3
        if row > 1:
            return 3
        if row < 1:
            return 2
        return 1


class RandomWalker():
    def __init__(self, *args, **kwargs):
        pass
    
    def select_action(self, obs):
        return np.random.choice([0, 1, 2, 3])
    

def collect_data(env, agent, num_episodes):
    obss = []
    actions = []
    next_obss = []
    rewards = []
    terminals = []
    returns = []
    traj_returns = []
    
    for e in range(num_episodes):
        obs = env.reset()
        done = False
        traj_return = 0
        this_rewards = []
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            onehot_action = np.zeros(4, dtype=np.float32)
            onehot_action[action] = 1.0
            traj_return += reward
            
            obss.append(obs)
            actions.append(onehot_action)
            next_obss.append(next_obs)
            rewards.append(reward)
            this_rewards.append(reward)   # this reward is used for returns
            terminals.append(done)
            
            obs = next_obs
        for i in range(len(this_rewards)-2, -1, -1):
            this_rewards[i] = this_rewards[i] + this_rewards[i+1]
        returns.extend(this_rewards)
        traj_returns.append(traj_return)
        
    obss = np.stack(obss, axis=0).astype(np.float32)
    actions = np.asarray(actions).astype(np.float32)
    next_obss = np.stack(next_obss, axis=0).astype(np.float32)
    rewards = np.asarray(rewards).astype(np.float32)
    terminals = np.asarray(terminals).astype(np.float32)
    returns = np.asarray(returns).astype(np.float32)
    traj_returns = np.asarray(traj_returns).astype(np.float32)
    return obss, actions, next_obss, rewards, terminals, returns, traj_returns


def get_cliff_dataset(config={
    "random": 10000, 
    # "blind0": 50, 
    # "blind1": 50,
    # "blind2": 50, 
    # "optimal": 50
}):
    np.random.seed(42)
    env = CliffWalking()
    agents = {
        "random": RandomWalker(), 
        "blind0": BlindWalker(0), 
        "blind1": BlindWalker(1), 
        "blind2": BlindWalker(2), 
        "optimal": OptimalWalker()
    }
    all_obss = []
    all_actions = []
    all_next_obss = []
    all_rewards = []
    all_terminals = []
    all_returns = []
    
    for cls, episode_num in config.items():
        if episode_num == 0:
            continue
        agent = agents[cls]
        obss, actions, next_obss, rewards, terminals, returns, traj_returns = collect_data(env, agent, episode_num)
        all_obss.append(obss)
        all_actions.append(actions)
        all_next_obss.append(next_obss)
        all_rewards.append(rewards)
        all_terminals.append(terminals)
        all_returns.append(returns)
    all_obss = np.concatenate(all_obss, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_next_obss = np.concatenate(all_next_obss, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    all_terminals = np.concatenate(all_terminals, axis=0)
    all_returns = np.concatenate(all_returns, axis=0)
    return env, {
        "observations": all_obss, 
        "actions": all_actions, 
        "next_observations": all_next_obss, 
        "rewards": all_rewards, 
        "terminals": all_terminals, 
        "returns": all_returns, 
        "ends": all_terminals.copy(), 
    }



if __name__ == "__main__":
#     env = CliffWalking()
#     agent = OptimalWalker(2)
#     obss, actions, next_obss, rewards, terminals, _, traj_returns = collect_data(env, agent, 1000)
#     print(
# f"""
# Traj returns: {traj_returns}
# return max: {traj_returns.max()}
# return min: {traj_returns.min()}
# return average: {traj_returns.mean()}
# """
# )
    env, dataset = get_cliff_dataset({
        "random": 10000
    })
    optimal_actions = np.zeros([3, 5], dtype=object)
    for x in range(5):
        for y in range(3):
            _filter = (dataset["observations"][:, 0] == x) & (dataset["observations"][:, 1] == y)
            rtgs = dataset["returns"][_filter]
            actions = dataset["actions"][_filter]
            if len(rtgs) == 0:
                continue
            max_rtg = rtgs.max()
            optimal_action = actions[rtgs == max_rtg].mean(axis=0)
            optimal_action = np.arange(4)[optimal_action != 0]
            # optimal_action = np.argmax(actions[rtgs == max_rtg].mean(axis=0))
            optimal_actions[2-y, x] = optimal_action
    print(optimal_actions)
    