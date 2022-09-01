import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_result(save_epi_reward):
        plt.plot(save_epi_reward)
        plt.show()

def save_fig_csv(path, n_episodes, rewards, max_delays):
        rewards = np.squeeze(rewards)
        max_delays = np.squeeze(max_delays)
        plt.plot(rewards)
        plt.xlim(0, n_episodes)
        plt.savefig('./result/'+path+'/reward.png')
        plt.clf()
        plt.plot(max_delays)
        plt.xlim(0, n_episodes)
        plt.savefig('./result/'+path+'/max_delay.png')
        plt.clf()
        average_rewards = np.convolve(rewards, np.ones(100), 'valid') / 100
        plt.plot(average_rewards)
        plt.xlim(0, n_episodes)
        plt.savefig('./result/'+path+'/average_reward.png')
        plt.clf()

        df = pd.DataFrame({'reward': rewards, 'max_delay': max_delays})
        df.to_csv('./result/'+path+'/train.csv')