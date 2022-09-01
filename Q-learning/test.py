import numpy as np
import pandas as pd

from env import PortBased0
from multi import MultiAgent

def test(max_episode_num, n_agents, topology, node_type, random, path):
    max_episode_num = max_episode_num
    n_agents = n_agents
    topology = topology
    node_type = node_type
    random = random
    path = path

    env = PortBased0(n_agents=n_agents, topology=topology,
                     node_type=node_type, random=random)

    if n_agents == 1:
        state_dim = [sum(env.state_dim)]
        action_bound = [2**int(state_dim[0]/8)]
        action_dim = action_bound
    else:
        state_dim = env.state_dim
        action_bound = env.action_bound
        action_dim = action_bound
    
    multi = MultiAgent(n_agents=n_agents, state_dim=state_dim,
                        action_dim=action_dim)

    if node_type == 0:
        print(path)
        multi.load_checkpoint('./result/'+path+'/model/')

    save_epi_reward = []
    max_delays = []
    for ep in range(10):
        time = 0
        epi_reward = 0
        state = env.reset()

        while True:
            if n_agents == 1:
                    state = np.reshape(state, (1,1,-1))
            action = multi.choose_action(state, True)
            state, reward, done, _ = env.step(action)
            time += 1

            print('Time: ', time, 'Reward: ', reward)
            epi_reward+=sum(reward)

            if all(done) or time==100:
                break
        
        max_delay = env.get_max_delay()
        print('Episode: ', ep, 'Reward: ', epi_reward, 'Max Delay: ', max_delay)
        save_epi_reward.append(epi_reward)
        max_delays.append(max_delay)

    df = pd.DataFrame({'reward': np.squeeze(save_epi_reward), 
                        'max_delay': np.squeeze(max_delays)})
    df.to_csv('./result/'+path+'/test.csv')

def get_file_name(max_episode_num, n_agents, topology, node_type, random, f_name):
    topology = '_topology' + str(topology)
    episodes = '_' + str(max_episode_num) + 'eps'
    if random:
        random = '_random'
    else:
        random = '_fix'

    if not node_type:
        if n_agents == 1:
            return f_name + '_sarl' + topology + random + episodes
        else:
            return f_name + '_marl' + topology + random + episodes
    elif node_type == 1:
        return 'fifo' + topology + random + episodes 
    elif node_type == 2:
        return 'rr' + topology + random + episodes
    elif node_type == 3:
        return 'ha' + topology + random + episodes

if __name__=="__main__":
    max_episode_num = 1000
    n_agents = 1
    topology = 0
    node_type = 0
    random = 0
    f_name= 'ddqn'

    test(max_episode_num=max_episode_num, n_agents=n_agents, topology=topology, 
            node_type=node_type, random=random, f_name=f_name)