import math
import random
from collections import deque 
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from agent_dir.agent import Agent
import scipy
import scipy.misc
import numpy as np

FloatTensor = torch.cuda.FloatTensor 
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor
Tensor = FloatTensor

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.linear1 = nn.Linear(32*8*8, 128)
        self.linear2 = nn.Linear(128,3)
        self.softmax = nn.Softmax()
        self.reward_list = []
        self.log_prob_list = []
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.linear1(x.view(x.size(0),-1)))
        return self.softmax(self.linear2(x))
    

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        self.gamma = 0.99
        print("Building PG  model...")
        self.model = Policy()
        print("Building optimizer ...")
        self.opt = optim.RMSprop(self.model.parameters(),lr = 1e-4, weight_decay = 0.99)
        self.save_every = 50000
        self.reward_queue = deque([])
        self.prev_state = None
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            checkpoint = torch.load('./save/pg/15100000.tar')
            self.model = Policy()
            self.model.load_state_dict(checkpoint['policy'])
            self.model.eval()
            print("Latest reward of 30 episode: {}".format(checkpoint['reward_queue']))
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def optimize_model(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.model.reward_list[::-1]:
            if r != 0:
                R = 0
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for log_prob, reward in zip(self.model.log_prob_list, rewards):
            policy_loss.append(-log_prob * reward)

        self.opt.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.opt.step()

        del self.model.reward_list[:] 
        del self.model.log_prob_list[:]

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        time_step = 0 
        for i_episode in count():
            raw_current_state = self.env.reset()
            epi_reward = 0
            while True:
                time_step += 1
                action = self.make_action(raw_current_state)
                raw_current_state, reward, done,_ = self.env.step(action)
                self.model.reward_list.append(reward)
                epi_reward += reward
                if (time_step+1)%self.save_every == 0:
                    torch.save({'policy':self.model.state_dict(),'opt':self.opt.state_dict(),'reward_queue':self.reward_queue},
                                './save/pg/{}.tar'.format(time_step+1)) 
                if done:
                   self.reward_queue.append(epi_reward)
                   if len(self.reward_queue) > 30:
                       self.reward_queue.popleft()
                   break            

            # Perform one step of the optimization (on the target network)
            print("Finish episode {}, cumulate time step {}, score {}, updating model...".format(i_episode+1,time_step+1, epi_reward))
            self.optimize_model()
           
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        current_state = prepro(observation)
        diff_state = current_state - self.prev_state if self.prev_state is not None else np.zeros((80,80))  
        self.prev_state = current_state           
        state = torch.from_numpy(diff_state).float().view(1,1,80,80)
        probs = self.model(Variable(state))
        sample_action = probs.multinomial()
        action = sample_action.data[0,0]+1
        self.model.log_prob_list.append(probs[0,sample_action.data[0,0]].log())
        return action

