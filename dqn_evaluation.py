#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:07:22 2018

@author: anand
"""
import gym
import torch
from torch.autograd import Variable
import random
from dqn import get_screen
from dqn import queue
from itertools import count

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

# use eps-greedy policy eps=0.05 for evaluating trained agent
episode_score = []
n_episodes = 100

# load trained agent
model = torch.load('DQN_agent.pt')

model.cuda()
# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')

# number of allowed actions for env
n_valid_actions = env.action_space.n

def eval_policy(state):
	global steps_done
	# convert state to FloatTensor and put it on gpu
	state = state.type(FloatTensor) / 255
	state = state.cuda()
	
	sample = random.random()
	
	# setting eps param of greedy policy
	eps_threshold = 0.05	
	if sample > eps_threshold:
		return model(Variable(state, volatile=True)).data.max(1)[1].view(1, 1)
	else:
		return LongTensor([[random.randrange(n_valid_actions)]])


# begin eval episodes
for i in range(n_episodes):
	# Init env
	frame = env.reset()
	observation = get_screen(frame)
	observation = observation
	
	# init frame_history
	frame_history = queue()
	
	# setting intial game state = (frame_0, frame_0, frame_0, frame_0)
	for j in range(4):
		frame_history.push(observation)
	
	state = frame_history.get_tensor()
	episode_reward = 0
	for t in count():
		# perform action using policy
		action = eval_policy(state)
		action = action.cpu()
		
		observation, reward, done, _ = env.step(action) #action[0,0]
		# preprocess new game frame
		observation = get_screen(observation)
		observation = observation.squeeze(0)
		
		# Render
		#env.render()
		
		episode_reward = episode_reward + reward
		#reward = Tensor([reward])
		
		if not done:
			frame_history.pop()
			frame_history.push(observation)
			next_state = frame_history.get_tensor() 
			
		else:
			next_state = None
		
		# Go to next state
		state = next_state
		if done:
			episode_score.append(episode_reward)
			# print score obtained in current episode
			print('Episode {} Total Score: {}'.format(i,episode_reward))
			break
	
env.render(close=True)
		