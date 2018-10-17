#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:12:47 2018

@author: anand
"""
import torch
import random
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition',('state','action','next_state','reward'))
class RingBuf:
	def __init__(self, size):
		# Pro-tip: when implementing a ring buffer, always allocate one extra element,
		# this way, self.start == self.end always means the buffer is EMPTY, whereas
		# if you allocate exactly the right number of elements, it could also mean
		# the buffer is full. This greatly simplifies the rest of the code.
		self.data = [None] * (size + 1)
		self.start = 0
		self.end = 0
		self.size = size
		         
	def push(self, *element):
		self.data[self.end] = Transition(*element)
		self.end = (self.end + 1) % len(self.data)
		# end == start and yet we just added one element. This means the buffer has one
		# too many element. Remove the first element by incrementing start.
		if self.end == self.start:
			self.start = (self.start + 1) % len(self.data)
        
	def __getitem__(self, idx):
		return self.data[(self.start + idx) % len(self.data)]
	    
	def __len__(self):
		if self.end < self.start:
			return self.end + len(self.data) - self.start
		else:
			return self.end - self.start
	        
	def __iter__(self):
		for i in range(len(self)):
			yield self[i]
				
	def sample(self, batch_size):
		
		idxs = random.sample(range(3, self.__len__() - 2), batch_size)
		state_batch = []
		action_batch = []
		reward_batch = []
		next_state_batch = []
		for i in idxs:
			# frame of reference
			curr_data = self.data[i]
			# past frames
			prev_data = self.data[i - 1]
			pprev_data = self.data[i - 2]
			
			# future frames
			next_data = self.data[i + 1]
			nnext_data = self.data[i + 2]
			state_batch.append(torch.cat((pprev_data.state, prev_data.state, curr_data.state, curr_data.next_state), 0).unsqueeze(0))
			action_batch.append(curr_data.action)
			reward_batch.append(curr_data.reward)
			next_state_batch.append(torch.cat((curr_data.state, curr_data.next_state, next_data.next_state, nnext_data.next_state), 0).unsqueeze(0))
			
		return Transition(state_batch, action_batch, next_state_batch, reward_batch)