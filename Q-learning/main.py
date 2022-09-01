import argparse
from platform import node
from random import random
import numpy as np
import time
import os

from utils import save_fig_csv
from env import PortBased0
from multi import MultiAgent
from test import test

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--actor_lr', type=float, default=1e-3)
parser.add_argument('--critic_lr', type=float, default=1e-4)
parser.add_argument('--clip_ratio', type=float, default=0.1)
parser.add_argument('--lmbda', type=float, default=0.95)
#single lr: 1e-3, 1e-4
#multi lr: 1e-4, 1e-4
#lr은 일단 수렴하는 수준까지 높인 뒤에 서서히 낮추자

args = parser.parse_args()

def train(max_episode_num, n_agents, topology, node_type, random, batch_size, path):
    total_start_time = time.time()

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
                        action_dim=action_dim, batch_size=batch_size)
    save_epi_reward = []
    max_delays = []
    state_action_log = []

    for ep in range(int(max_episode_num)):
        start = time.time()

        step, episode_reward = 0, 0
        epi_done = False
        state = env.reset()

        while not epi_done:
            step+=1
            if n_agents == 1:
                state = np.reshape(state, (1,1,-1))

            action = multi.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            if step==5: done = np.array([1 for i in done])
            if all(done): epi_done=True

            train_reward = (reward)#/ 10.0  # <-- normalization

            #print(state, action, train_reward, next_state, done)
            state_action_log.append([state, action, train_reward, next_state, done])
            
            multi.learn(state, action, train_reward, next_state, done)

            # update current state
            state = next_state
            episode_reward += sum(reward)
        #multi.agents[0].print_lr()

        end = time.time()


        ## display rewards every episode
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        hours_, rem_ = divmod(end-total_start_time, 3600)
        minutes_, seconds_ = divmod(rem_, 60)
        print('Episode: ', ep+1, 'Step: ', step, 'Reward: ', episode_reward,
             'Episode time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds),
             'Elapsed time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours_),int(minutes_),seconds_))

        save_epi_reward.append(episode_reward)
        max_delay = env.get_max_delay()
        max_delays.append(max_delay)


        ## save weights every episode
        #multi.save_checkpoint('./result/'+path+'/model/')
        multi.epsilon_decay()

        if episode_reward==100.0:
            print(state_action_log)
        else: state_action_log = []


    #print(agent.save_epi_reward)
    total_end_time = time.time()
    hours, rem = divmod(total_end_time-total_start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Average reward: ', sum(save_epi_reward[-100:])/100,
        'Max reward: ', max(save_epi_reward),
        'Max delay: ', max_delay,
        'Total elapsed time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))
    save_fig_csv(path, max_episode_num, save_epi_reward, max_delays)

def make_folder_name(n_agents, topology, node_type, random, f_name):
    topology = 'topology' + str(topology)
    if random:
        random = '/random_'
    else:
        random = '/fix_'

    if not node_type:
        if n_agents == 1:
            return topology + '/dqn/sarl' + random + f_name
        else:
            return topology + '/dqn/marl' + random + f_name
    elif node_type == 1:
        return topology + '/fifo' + random + f_name
    elif node_type == 2:
        return topology + '/rr' + random + f_name
    elif node_type == 3:
        return topology + '/ha' + random + f_name
    elif node_type == 4:
        return topology + '/n_fifo' + random + f_name

if __name__ == '__main__':
    max_episode_num = 3000
    #n_agents는 topology에 rl_node와 동일하게 하면 multi agent임
    n_agents = 1
    topology = 3
    node_type = 0
    random = 0
    batch_size = 64

    for i in range(0,1):
        f_name = str(max_episode_num) + 'eps_' + str(i)

        path = make_folder_name(n_agents=n_agents,
                topology=topology, node_type=node_type, random=random,
                f_name=f_name)
        
        if not os.path.exists('./result/' + path):
            os.makedirs('./result/' + path)
        if not os.path.exists('./result/' + path + '/model'):
            os.makedirs('./result/' + path + '/model')

        print('Train ', i)
        train(max_episode_num=max_episode_num, n_agents=n_agents, topology=topology, 
                node_type=node_type, random=random, batch_size=batch_size, path=path)

        #print('Test ', i)
        #test(max_episode_num=max_episode_num, n_agents=n_agents, topology=topology, 
        #        node_type=node_type, random=random, path=path)
