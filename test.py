# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')

obs = env.reset()
frame, reward, is_done, _ = env.step(env.action_space.sample())

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
#process_obs = get_screen(frame)
process_obs = get_screen(frame)
process_obs = process_obs.numpy().transpose(1,2,0)
process_obs = np.ascontiguousarray(process_obs*255, dtype=np.uint8)

screen_tensor = torch.from_numpy(process_obs)
#process_obs = process_obs*255

#view processed game screen
plt.figure()
plt.imshow(process_obs, interpolation='none')
plt.title('Example extracted screen')
plt.show()
'''
observation = get_screen(frame)
observation = observation.squeeze(0)
observation.size()
