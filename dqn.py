#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:44:26 2018

@author: anand
"""
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from RingBuf import RingBuf

import gym.wrappers as wrappers
import gym.spaces
from atari_wrappers import *
import dqn_utils
import itertools 

import pickle
import sys
import os

import dqn_agent
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# defining seprate handles for cpu & gpu tensors to save gpu memory
cudaFloatTensor = torch.cuda.FloatTensor
FloatTensor = torch.FloatTensor

cudaLongTensor = torch.cuda.LongTensor
LongTensor = torch.LongTensor

cudaIntTensor = torch.cuda.IntTensor
IntTensor = torch.IntTensor

cudaByteTensor = torch.cuda.ByteTensor
ByteTensor = torch.ByteTensor

cudaTensor = cudaFloatTensor
Tensor = FloatTensor


# Create a Pong environment
env = gym.make('Pong-v4')
save_dir = "dqn_vid/"
env = wrappers.Monitor(env, save_dir, video_callable=False, force=True)
# using DeepMind Atari settings
env = wrap_deepmind(env)

# number of allowed actions for env
n_valid_actions = env.action_space.n

class DQN(nn.Module):
	
	def __init__(self, input_shape=(4,84,84)):
		# architecture as specified in "Human-Level Control through Deep Reinforcement Learning"
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1)
		
		flatten_size = self._get_conv_output(input_shape)
		
		self.fc1 = nn.Linear(in_features = flatten_size, out_features = 512)
		self.output = nn.Linear(in_features = 512, out_features = n_valid_actions)

	
	# generate random input sample and forward-pass through conv-layers to infer shape
	def _get_conv_output(self, shape):
		bs = 1
		input = Variable(torch.rand(bs, *shape))
		output_feat = self._forward_features(input)
		n_size = output_feat.data.view(bs, -1).size(1)
		return n_size

	def _forward_features(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		return x
	
	def forward(self, x):
		x = self._forward_features(x)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = self.output(x)
		return x
	
	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


# Define Training Parameters and Init Agent
batch_size = 32
init_replay_size = 50000 # use random policy to fill 50000 frames of replay buffer initially before Q-learning updates
gamma = 0.99
eps_start = 1.0
eps_end = 0.1
decay_param = 1000000 # linear decay of eps until decay_param steps then fixed at 0.1 
steps_done = 0

# init replay memory config params
replay_memory_size = 1000000 # size of memory buffer (in frames)
frame_history_len = 4 # number of stacked frames given as input to agent

# init agent and replay memory
model = DQN()
target_model = DQN()
replay_memory = dqn_utils.ReplayBuffer(replay_memory_size, frame_history_len) 

# function to clone Q-network to create Target Network Q_hat
def clone_model(model):
	torch.save(model,'Q_network_model.pt')
	target_model = torch.load('Q_network_model.pt')
	return target_model

if torch.cuda.is_available():
	model.cuda()
	target_model.cuda()

optimizer = optim.RMSprop(model.parameters(), alpha = 0.95, eps = 0.01, lr=0.00025, centered=True) 

def eps_greedy_policy(state):
	global steps_done
	# change dims to BCHW
	#print(state.shape)
	state = np.transpose(state, (2, 0, 1))
	# convert state to FloatTensor and put it on gpu
	state = state.astype(np.float32) / 255
	state = torch.from_numpy(state)
	state = state.cuda()
	
	sample = random.random()
	if steps_done <= decay_param:
		eps_threshold = (-0.9*steps_done)/decay_param + eps_start
	else:
		eps_threshold = eps_end
		
	if sample > eps_threshold:
		return model(Variable(state, volatile=True)).data.max(1)[1].view(1, 1)
	else:
		return LongTensor([[random.randrange(n_valid_actions)]])

# function which performs q-learning 
def q_learning():
	global last_sync
	
	state_batch, action_batch, reward_batch, next_state_batch, done_mask = replay_memory.sample(batch_size)
	
 	# Changing dims to BCHW
	state_batch = np.transpose(state_batch, (0, 3, 1, 2)) 
	next_state_batch = np.transpose(state_batch, (0, 3, 1, 2))
	
	next_state_batch = next_state_batch.astype(np.float32) / 255
	next_state_batch = torch.from_numpy(next_state_batch)
	next_state_batch = Variable(next_state_batch.cuda())
	
	#convert it to FloatTensor before further processing by NN model
	state_batch = state_batch.astype(np.float32) / 255
	state_batch = torch.from_numpy(state_batch)
	state_batch = Variable(state_batch.cuda())
	
	# make them torch Tensors and put it on gpu
	action_batch = torch.from_numpy(action_batch).type(torch.LongTensor)
	action_batch = Variable(action_batch.cuda())
	
	reward_batch = torch.from_numpy(reward_batch) 
	reward_batch = Variable(reward_batch.cuda())
	
	# Compute Q(s_t,a)
	state_action_value = model(state_batch).gather(1,action_batch)
	
	# Compute V(s') i.e. value of next state using copy of agent model i.e. target model
	#next_state_value = Variable(torch.zeros(batch_size).type(Tensor))
	next_state_value = Variable(torch.zeros(batch_size).type(cudaTensor))
	
	# put indexing on gpu
	done_mask = done_mask.type(cudaByteTensor)
	next_state_value[done_mask] = target_model(next_state_batch).max(1)[0]
	next_state_value.volatile = False
	
	# Compute expected Q values i.e. r + gamma*V(s')
	expected_state_action_value = (next_state_value*gamma) + reward_batch
	
	# Compute  Huber loss
	loss = F.smooth_l1_loss(state_action_value, expected_state_action_value, False)
	
	# perform 1 step of gradient descent in TD error direction
	optimizer.zero_grad()
	loss.backward()
	
	# gradient clipping
	for param in model.parameters():
            param.grad.data.clamp_(-5, 5)
	optimizer.step()
	
# Training Loop
#n_episodes = 5000
frame_count = 0 # used to keep track of total game frames played to schedule a target network update
update_cycle = 10000
max_steps = 2000000 # for Pong = 2M, for Breakout and other harder games = 10-50M
episode_score = []
learning_freq = 4
episode_count = 0

# Init Environment and states
last_obs = env.reset()
print(last_obs.shape)
# init loggging variables
t_log = []
mean_reward_log = []
best_mean_log = []
episodes_log = []
mean_episode_reward = -float('nan')
best_mean_episode_reward = -float('inf')

# begin training
for t in itertools.count():
	
	# break out if agent trained for > max_steps # weight updates
	if steps_done > max_steps:
            break
	
	idx = replay_memory.store_frame(last_obs)
	
	if t <= init_replay_size:
		action = env.action_space.sample() # random actions at the start to fill up buffer
	
	else:
		state = replay_memory.encode_recent_observation()
		# perform action using policy
		action = eps_greedy_policy(state)
	
	action = action.cpu()
	last_obs, reward, done, info = env.step(action) #action[0,0]
	
	# storing transition in memory buffer
	replay_memory.store_effect(idx, action, reward, done)
	reward = Tensor([reward])
			
	if done == True:
		last_obs = env.reset()
		done = False		
	
	# perform learning only after intial 50000 frames fills up in buffer using random policy
	if (t > init_replay_size and t % learning_freq == 0 and replay_memory.can_sample(batch_size)):						
		# 1 step grad descent on agent network
		q_learning()
		steps_done = steps_done + 1 # count of weight updates
		
		if t % update_cycle == 0:
			target_model = clone_model(model)
			target_model.cuda()
			print("\nTarget Network updated\n")	
	# Logging results
	episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()	
	if len(episode_rewards) > 0:
		mean_episode_reward = np.mean(episode_rewards[-100:])
		if len(episode_rewards) > 100:
			best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
		if t % update_cycle == 0 and t > init_replay_size:
			print("Timestep %d" % (t,))
			t_log.append(t)
			print("Mean reward over (100 episodes) %f" % mean_episode_reward)
			mean_reward_log.append(mean_episode_reward)
			print("Best mean reward %f" % best_mean_episode_reward)
			best_mean_log.append(best_mean_episode_reward)
			print("Episodes %d" % len(episode_rewards))
			episodes_log.append(len(episode_rewards))
			sys.stdout.flush()
		
		if t % 2*update_cycle == 0 and t > init_replay_size:
			# save all logs to be used for plotting later
			training_log = ({'t_log': t_log, 'mean_reward_log': mean_reward_log, 'best_mean_log': best_mean_log, 'episodes_log': episodes_log })
			filename = "dqn_logs/" + str(t) + "_" + "train_logs.pkl"
			with open(filename, "wb") as fp:
				pickle.dump(training_log, fp) 


env.render(close=True)
# save model at end of training 
torch.save(model,'DQN_agent.pt')
# save replay buffer if we want to resume training in the future
torch.save(replay_memory, 'memory.pt')

