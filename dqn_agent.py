#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 03:09:21 2018

@author: anand
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Build DQN Agent (function approximation for Q(s,a); used to make both target network and current network)
class DQN(nn.Module):
	
	def __init__(self, n_valid_actions, input_shape=(4,84,84)):
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