import numpy as np
import tensorflow as tf
from tensorflow import keras

class ReplayBuffer():
	def __init__(self, max_size, input_dims):
		self.memory_size = memory_size
		self.memory_counter = 0

		self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)