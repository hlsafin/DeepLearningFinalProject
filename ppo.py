import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, 64)
        self.output = nn.Linear(64, num_action)
        #self.optimizer = optim.RMSprop(self.parameters(), lr=0.001)
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.0008)
        self.tanh = nn.LeakyReLU()
        # nn.LeakyReLU
        self.apply(self.init_)

    def forward(self, x):
        x = self.tanh(self.fc1(x))

        x = self.tanh(self.fc2(x))

        actionProb = F.softmax(self.output(x), dim=1)
        return actionProb

    def init_(self, m):
        if isinstance(m, nn.Linear):
            gain = torch.nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_normal_(m.weight, gain)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, 64)
        self.state_value = nn.Linear(64, 1)
        #self.optimizer = optim.RMSprop(self.parameters(), lr=0.004)
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.002)
        self.tanh = nn.LeakyReLU()
        self.apply(self.init_)

        # self.norm = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        # x = self.norm(x)
        x = self.tanh(self.fc2(x))

        value = self.state_value(x)
        return value

    def init_(self, m):
        if isinstance(m, nn.Linear):
            gain = torch.nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_normal_(m.weight, gain)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


gamma = 0.99
seed = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')
# env = gym.make('Pong-ram-v0')
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
env.seed(seed)
reward_list_plot = []
Trans_list = []
clipParam = 0.2
maxGrad = 0.5
ppo_update_time = 10
batch_size = 32
actorNet = Actor().to(device)
criticNet = Critic().to(device)
transitions = []
counter = 0
training_step = 0
plot_raw_data=[]
score = 0.0
score2=0
epoch_length = 10000
done = False


def normalize(t, eps=1e-5):
    return (t - t.mean()) / (t.std() + eps)


def updates(net, loss):
    net.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), maxGrad)
    net.optimizer.step()


for epoch_num in range(epoch_length):
    state = env.reset()

    while not done:
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            actionProb = actorNet(state.to(device))
        Cate = Categorical(actionProb)
        action = Cate.sample()
        action = action.item()
        actionProb = actionProb[:, action].item()
        nextState, reward, done, _ = env.step(action)
        trans = [state, action, actionProb, reward, nextState]

        transitions.append(trans)

        #### store transitions to buffer ####

        state = nextState
        score += reward

        if done:
            done = False

            reward_list_plot.append(score)
            score = 0.0


            # score = 0

            if len(transitions) >= batch_size:

                ### update happens here ###########

                state = torch.vstack([x[0] for x in transitions])

                reward = [x[3] for x in transitions]
                action = torch.tensor([x[1] for x in transitions], dtype=torch.long).view(-1, 1).to(device)
                oldActionProb = torch.tensor([x[2] for x in transitions], dtype=torch.float).view(
                    -1, 1).to(device)
                R = 0
                R_list = []
                for r in reward[::-1]:
                    R = r + gamma * R
                    R_list.insert(0, R)
                R_list = torch.tensor(R_list, dtype=torch.float)
                for k_epoch in range(ppo_update_time):

                    for x in range(5):
                        idx = list(np.random.randint(0, len(transitions), int(len(transitions) * 0.75)))

                        R_idx = R_list[idx].view(-1, 1)
                        V = criticNet(state[idx].to(device))
                        delta = R_idx.to(device) - V
                        delta = normalize(delta)
                        advantage = delta.detach()

                        actionProb = actorNet(state[idx].to(device)).gather(1, action[
                            idx].to(device))

                        ratio = (actionProb / oldActionProb[idx])
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1 - clipParam, 1 + clipParam) * advantage
                        actionLoss = -torch.min(surr1, surr2).mean()

                        valueLoss = F.mse_loss(R_idx.to(device), V.to(device))
                        ### updates #####
                        updates(criticNet, valueLoss)
                        updates(actorNet, actionLoss)

                        training_step += 1

                del transitions[:]

            if epoch_num % 20 == 0 and epoch_num != 0:
                print("# of episode :{}, avg score : {:.1f}".format(epoch_num, score / 20))

                #reward_list_plot.append(score/100)



                numbers_series = pd.Series(reward_list_plot)
                windows = numbers_series.rolling(20,closed='both')
                moving_averages = windows.mean()

                moving_averages_list = moving_averages.tolist()
                #without_nans = moving_averages_list[20 - 1:]
                #plot_raw_data.append(moving_averages_list)

                plt.plot(reward_list_plot)
                plt.plot(moving_averages_list)
                plt.legend(["Raw Score Per Episode ", "Moving Average Score"], loc="lower right")
                plt.xlabel('Episode (PPO)')
                plt.ylabel('Episode Score')

                plt.show()
                #score = 0.0
            break
