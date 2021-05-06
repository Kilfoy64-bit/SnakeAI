try:
	import sys
	import random
	from collections import deque	

	from objects import SnakeGameAI, Direction, Point
	from model import DQN
	from helper import plot

	import torch
	import torch.optim as optim
	import torch.nn as nn

	import cv2
	import numpy as np

except ImportError as err:
    print (f"couldn't load module. {err}")
    sys.exit(2)

# The Agent class plays the SankeGame, but does not continue training the model.
class Agent:

	BLOCK_SIZE = 20

	def __init__(self, trainedModelPath = None):
		self.game = SnakeGameAI()
		self.episodes = 0
		self.model = DQN(11, 1024, 3)
		if trainedModelPath is not None:
			self.model.load_state_dict(torch.load(trainedModelPath))
		
	def get_state(self):

		# pixel_array = self.game.get_game_frame() 
		# pixel_array = cv2.resize(pixel_array, (40,30), interpolation=cv2.INTER_CUBIC) # resizes the pixel array
		# pixel_array = np.mean(pixel_array, axis=2) # converts the image to grayscale
		# pixel_array = pixel_array/255.0 # Normalizes the values in the pixel array
		# pixel_array = pixel_array.flatten() # Flattens the pixel array to a single axis

		head = self.game.snake.body[0].getPoint()
		food = self.game.food.getPoint()

		left_point = Point(head.x - self.BLOCK_SIZE, head.y)
		right_point = Point(head.x + self.BLOCK_SIZE, head.y)
		upwards_point = Point(head.x, head.y - self.BLOCK_SIZE)
		downwards_point = Point(head.x, head.y + self.BLOCK_SIZE)

		left_direction = self.game.snake.direction == Direction.LEFT
		right_direction = self.game.snake.direction == Direction.RIGHT
		upwards_direction = self.game.snake.direction == Direction.UP
		downwards_direction = self.game.snake.direction == Direction.DOWN

		game_info = [
			# Danger straight
			(right_direction and self.game.is_collision(right_point)) or
			(left_direction and self.game.is_collision(left_point)) or
			(upwards_direction and self.game.is_collision(upwards_point)) or
			(downwards_direction and self.game.is_collision(downwards_point)),
			
			# Danger right
			(right_direction and self.game.is_collision(downwards_point)) or
			(left_direction and self.game.is_collision(upwards_point)) or
			(upwards_direction and self.game.is_collision(right_point)) or
			(downwards_direction and self.game.is_collision(left_point)),

			# Danger left
			(right_direction and self.game.is_collision(upwards_point)) or
			(left_direction and self.game.is_collision(downwards_point)) or
			(upwards_direction and self.game.is_collision(left_point)) or
			(downwards_direction and self.game.is_collision(right_point)),

			# Move Direction
			left_direction,
			right_direction,
			upwards_direction,
			downwards_direction,

			# Food location
			food.x < head.x, # food is left of the head
			food.x > head.x, # food is right of the head
			food.y < head.y, # food is above the head
			food.y > head.y  # food is below the head
		]

		# state = np.concatenate((game_info, pixel_array), axis=None)
		state = game_info
		return np.array(state, dtype=float)

	def select_action(self, state):
		final_move = [0,0,0]
		state0 = torch.tensor(state, dtype=torch.float)
		prediction = self.model(state0)
		move = torch.argmax(prediction)
		final_move[move] = 1
		
		return final_move

	def play(self):
		record = 0
		# Game loop
		while True:
			# get old state
			state_old = self.get_state()

			# get move
			final_move = self.select_action(state_old)

			# perform move and get new state
			_, game_over, score = self.game.play_step(final_move)

			if game_over:
				# train long memory
				self.game.reset()
				self.episodes += 1

				if score > record:
					record = score
				print(f"Game: {self.episodes}, Score: {score}, Record: {record}")

# The Trainer class adds functionality to the Agent model to allow learning
class Trainer(Agent):

	MAX_MEMORY = 100_100
	BATCH_SIZE = 128
	GAMMA = 0.9
	EPSILON = 80 # Decay rate is 1:1 with number of episodes (games played)
	LR = 0.001

	BLOCK_SIZE = 20

	def __init__(self, trainedModelPath = None):
		super().__init__(trainedModelPath)
		self.memory = deque(maxlen=self.MAX_MEMORY)
		if trainedModelPath is not None:
			self.model.load_state_dict(torch.load(trainedModelPath))
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
		self.criterion = nn.MSELoss()
		
	def remember(self, state, action, reward, next_state, game_over):
		self.memory.append((state, action, reward, next_state, game_over)) # pop left if MAX_MEMORY reached

	def train_long_memory(self):
		if len(self.memory) > self.BATCH_SIZE:
			mini_sample = random.sample(self.memory, self.BATCH_SIZE) # list of tuples
		else:
			mini_sample = self.memory
		
		states, actions, rewards, next_states, game_overs = zip(*mini_sample)
		self.train_step(states, actions, rewards, next_states, game_overs)

	def train_short_memory(self, state, action, reward, next_state, game_over):
		self.train_step(state, action, reward, next_state, game_over)

	def train_step(self, state, action, reward, next_state, game_over):
		state = torch.tensor(state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.float)
		reward = torch.tensor(reward, dtype=torch.float)

		if len(state.shape) == 1:
			state = torch.unsqueeze(state, 0)
			next_state = torch.unsqueeze(next_state, 0)
			action = torch.unsqueeze(action, 0)
			reward = torch.unsqueeze(reward, 0)
			game_over = (game_over, )
		
		prediction = self.model(state)
		target = prediction.clone()

		# Q_new = reward + gamma * max(next_predicted Q value)
		for i in range(len(game_over)):
			Q_new = reward[i]
			if not game_over[i]:
				Q_new = reward[i] + self.GAMMA * torch.max(self.model(next_state[i]))
			
			target[i][torch.argmax(action).item()] = Q_new
		
		self.optimizer.zero_grad()
		loss = self.criterion(target, prediction)
		loss.backward()

		self.optimizer.step()

	def select_action(self, state):
		# random moves: tradeoff exploration/ exploitation
		random_move_opportunity = self.EPSILON - self.episodes
		final_move = [0,0,0]
		if random.randint(0,200) < random_move_opportunity:
			move = random.randint(0, 2)
			final_move[move] = 1
		else:
			state0 = torch.tensor(state, dtype=torch.float)
			prediction = self.model(state0)
			move = torch.argmax(prediction)
			final_move[move] = 1
		
		return final_move

	def play(self):
		plot_scores = []
		plot_mean_scores = []
		total_score = 0
		record = 0
		# Training loop
		while True:
			# get old state
			state_old = self.get_state()

			final_move = self.select_action(state_old)

			reward, game_over, score = self.game.play_step(final_move)

			state_new = self.get_state()
			
			self.train_short_memory(state_old, final_move, reward, state_new, game_over)

			self.remember(state_old, final_move, reward, state_new, game_over)

			if game_over:
				self.game.reset()
				self.episodes += 1
				self.train_long_memory()

				if score > record:
					record = score
					self.model.save()
				print(f"Game: {self.episodes}, Score: {score}, Record: {record}")

				# Plot graph
				plot_scores.append(score)
				total_score += score
				mean_score = total_score / self.episodes
				plot_mean_scores.append(mean_score)
				plot(plot_scores, plot_mean_scores)