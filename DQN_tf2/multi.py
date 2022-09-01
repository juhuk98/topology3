from dqn_agent import DDQNAgent

class MultiAgent:
    def __init__(self, n_agents=1, state_dim=[16], action_dim=[4], batch_size=64):
        self.agents = []
        self.n_agents = n_agents
        for agent_idx in range(self.n_agents):
            self.agents.append(DDQNAgent(state_dim[agent_idx], action_dim[agent_idx], batch_size))

    def choose_action(self, raw_obs, test=False):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.get_action(raw_obs[agent_idx], test)
            actions.append(action)
        return actions

    def save_checkpoint(self, path):
        print('... saving checkpoint ...')
        for agent_idx, agent in enumerate(self.agents):
            agent.save_weights(path+str(agent_idx)+'_')

    def load_checkpoint(self, path):
        print('... loading checkpoint ...')
        for agent_idx, agent in enumerate(self.agents):
            agent.load_weights(path+str(agent_idx)+'_')

    def learn(self):
        for agent in self.agents:
            agent.learn()

    def store_memory(self, state, action, reward, state_, done):
        for agent_idx, agent in enumerate(self.agents):
            #print(agent_idx, state[agent_idx], action[agent_idx],
            #    reward[agent_idx], state_[agent_idx], done[agent_idx])
            if not done[agent_idx] or any(state[agent_idx][0]):
                agent.store_memory(state[agent_idx], action[agent_idx],
                        reward[agent_idx], state_[agent_idx], done[agent_idx])

    def epsilon_decay(self):
        for agent in self.agents:
            agent.epsilon_decay()

    def update_target_model(self):
        for agent in self.agents:
            agent.update_target_model()