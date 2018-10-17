#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:48:43 2018

@author: anand
"""

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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


# if gpu is to be used
#use_cuda = torch.cuda.is_available()
#FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
#LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
#IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
#ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
#Tensor = FloatTensor

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


# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')

# number of allowed actions for env
n_valid_actions = env.action_space.n

'''
# Reset it, returns the starting frame
frame = env.reset()
# Render
observations = env.render()

is_done = False
while not is_done:
  # Perform a random action, returns the new frame, reward and whether the game is over
  frame, reward, is_done, _ = env.step(env.action_space.sample())
  # Render
  env.render()
'''
# used to store and operate on stack of game frames used as input to DQN agent
class queue():	
	def __init__(self):
		self.arr = []
		
	def push(self, item):
		self.arr.append(item)
		
	def pop(self):
		self.arr = self.arr[1 :]
		
	def get_tensor(self):
		#print self.arr[0].type(), self.arr[1].type(), self.arr[2].type(), self.arr[3].type() 
		history = torch.cat((self.arr[0], self.arr[1], self.arr[2], self.arr[3]),  0)
		return history.unsqueeze(0)

# function to preprocess a frame  
def get_screen(observation):
	
	# render screen and change dims to torch format
	#screen = env.render(mode='rgb_array').transpose(2,0,1)
	
	# observation is 210, 160, 3 uint8 array
	#screen = observation
	
	# convert to greyscale and resize and store as unit8
	#screen = np.ascontiguousarray(screen, dtype=np.uint8)
	#screen = torch.from_numpy(screen) 
	
	preprocess = T.Compose([T.ToPILImage(), T.Resize((84,84)), T.Grayscale(num_output_channels=1), T.ToTensor()])
	
	#convert preprocessed frame back to unit8 format to save memory
	screen = preprocess(observation)
	screen = screen.numpy()
	screen = np.ascontiguousarray(screen*255, dtype=np.uint8)
	screen = torch.from_numpy(screen)
	
	# Resize, and add a batch dimension (BCHW)
	#return preprocess(observation).unsqueeze(0).type(Tensor)
	return screen.unsqueeze(0)
'''
# to view preprocessed image
plt.figure()
plt.imshow(processed_screen, interpolation='none')
plt.title('Example extracted screen')
plt.show()
'''

# Build Duelling DQN Agent (function approximation for Q(s,a); used to make both target network and current network)
class Duelling_DQN(nn.Module):
	
	def __init__(self, input_shape=(4,84,84)):
		# architecture as specified in "Duelling Network Architectures for Deep Reinforcement Learning"
		super(Duelling_DQN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1)
		
		flatten_size = self._get_conv_output(input_shape)
		
		# value stream
		self.value_fc1 = nn.Linear(in_features = flatten_size, out_features = 512)
		self.value_fc2 = nn.Linear(in_features = 512, out_features = 1)
		
		# advantage stream
		self.advantage_fc1 = nn.Linear(in_features = flatten_size, out_features = 512)
		self.advantage_fc2 = nn.Linear(in_features = 512, out_features = n_valid_actions)
	
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
		
		
		adv = F.relu(self.advantage_fc1(x))
		val = F.relu(self.value_fc1(x))
		
		adv = self.advantage_fc2(adv)
		val = self.value_fc2(val).expand(x.size(0), n_valid_actions)
		
		x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), n_valid_actions)
		return x
	
	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

# Build Experience Replay to store and sample past game moves
Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ExperienceReplay(object):
	
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0
	
	# function to store past game transitions
	def push(self, *args):
		#when space left in memory buffer
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		#when memory buffer is full replace earliest memories with new memories
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity
	
	# function to sample minibatch 
	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)
	
	def __len__(self):
		return len(self.memory)

# Define Training Parameters and Init Agent
batch_size = 32
init_replay_size = 50000 # use random policy to fill 50000 frames of replay buffer initially before Q-learning updates
gamma = 0.99
eps_start = 1.0
eps_end = 0.1
decay_param = 500000 # linear decay of eps until 1st 1million steps then fixed at 0.1 

# if restarting from saved model and replay memory
#model = torch.load('DQN_agent.pt')
#memory = torch.load('memory.pt')

# function to clone Q-network to create Target Network Q_hat
def clone_model(model):
	torch.save(model,'Duelling_Double_Q_network_model.pt')
	target_model = torch.load('Duelling_Double_Q_network_model.pt')
	return target_model

# init agent and replay memory
model = Duelling_DQN()
target_model = clone_model(model)
memory = ExperienceReplay(240000)

if torch.cuda.is_available():
	model.cuda()
	target_model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=0.00025, eps=0.01, momentum=0.95, centered=True) 

steps_done = 0

def eps_greedy_policy(state):
	global steps_done
	# convert state to FloatTensor and put it on gpu
	state = state.type(FloatTensor) / 255
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

# Double Q-Learning Function
def double_q_learning():
	global last_sync
	# init memory buffer
	if len(memory) < init_replay_size:
		return
	transitions = memory.sample(batch_size)
	# Transpose the batch
	batch = Transition(*zip(*transitions))
	non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,batch.next_state)))
	
	# creating batches of states, next_states, actions, rewards
	non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]),volatile=True)
	non_final_next_states = non_final_next_states.type(FloatTensor) / 255
	non_final_next_states = non_final_next_states.cuda()
	
	#convert it to FloatTensor before further processing by NN model
	batch_state = torch.cat(batch.state)
	batch_state = batch_state.type(FloatTensor) / 255
	batch_state = batch_state.cuda()
	#state_batch = Variable(torch.cat(batch.state))
	state_batch = Variable(batch_state)
	
	# put it on gpu
	action_batch = Variable(torch.cat(batch.action).cuda()) 
	reward_batch = Variable(torch.cat(batch.reward).cuda())
	
	# Compute Q(s_t,a)
	state_action_value = model(state_batch).gather(1,action_batch)
	
	# Compute V(s') i.e. value of next state using copy of agent model i.e. target model
	#next_state_value = Variable(torch.zeros(batch_size).type(Tensor))
	next_state_value = Variable(torch.zeros(batch_size).type(cudaTensor))
	
	# double dqn involves breaking down max over actions 
	# into action selection(by online network) + action evaluation(by offline target network)
	
	# action selection
	q_values_next = model(non_final_next_states)
	best_actions = torch.max(q_values_next, axis=1)
	
	# TO DO... need to rework the equations again
	
	# put indexing on gpu
	non_final_mask = non_final_mask.type(cudaByteTensor)
	next_state_value[non_final_mask] = target_model(non_final_next_states).max(1)[0]
	next_state_value.volatile = False
	
	# Compute expected Q values i.e. r + gamma*V(s')
	expected_state_action_value = (next_state_value*gamma) + reward_batch
	
	# Compute  Huber loss
	loss = F.smooth_l1_loss(state_action_value, expected_state_action_value)
	
	# perform 1 step of gradient descent in TD error direction
	optimizer.zero_grad()
	loss.backward()
	
	# gradient clipping
	for param in model.parameters():
          param.grad.data.clamp_(-10, 10)
	optimizer.step()
# Training Loop
n_episodes = 10000
frame_count = 0 # used to keep track of total game frames played to schedule a target network update
update_cycle = 10000
episode_score = []

# begin training
for i in range(n_episodes):
	# Init Environment and states
	frame = env.reset()
	observation = get_screen(frame)
	observation = observation.squeeze(0)
	# Render
	#env.render()
	episode_reward = 0 
	
	# queue of 4 frames
	frame_history = queue()
	
	# setting intial game state = (frame_0, frame_0, frame_0, frame_0)
	for j in range(4):
		frame_history.push(observation)
		
	state = frame_history.get_tensor()
	
	for t in count():
		# perform action using policy
		action = eps_greedy_policy(state)
		action = action.cpu()
		
		observation, reward, done, _ = env.step(action) #action[0,0]
		# Render
		env.render()
		
		episode_reward = episode_reward + reward
		reward = Tensor([reward])
		
		if not done:
			frame_history.pop()
			observation = get_screen(observation)
			observation = observation.squeeze(0)
			frame_history.push(observation)
			next_state = frame_history.get_tensor() 
		else:
			next_state = None
			
		# storing transition in memory buffer
		memory.push(state,action,next_state,reward)
		
		# Go to next state
		state = next_state
		
		if frame_count % update_cycle ==0:
			target_model = clone_model(model)
			
		# 1 step grad descent on target network
		double_q_learning()
		frame_count = frame_count + 1
		steps_done = steps_done + 1
		if done:
			episode_score.append(episode_reward)
			# print score obtained in current episode
			print('Episode {} Total Score: {}'.format(i,episode_reward))
			#print ('Frame Count {}'.format(frame_count))
			break
	
env.render(close=True)
# save model at end of training 
torch.save(model,'Duelling_Double_DQN_agent.pt')
# save replay buffer if we want to resume training in the future
torch.save(memory, 'memory.pt')