try:
	import sys
	import random
	from collections import deque	

	from objects import SnakeGameAI, Direction, Point
	from model import DQN, Trainer
	from helper import plot

	import torch
	import cv2
	import numpy as np

except ImportError as err:
    print (f"couldn't load module. {err}")
    sys.exit(2)

class Agent:

	MAX_MEMORY = 100_100
	BATCH_SIZE = 128
	GAMMA = 0.9
	EPSILON = 80 # Decay rate is 1:1 with number of episodes (games played)
	LR = 0.001

	BLOCK_SIZE = 20

	def __init__(self, trainedModelPath = None):
		self.game = SnakeGameAI()
		self.episodes = 0
		self.memory = deque(maxlen=self.MAX_MEMORY)
		self.model = DQN(11, 1024, 3)
		if trainedModelPath is not None:
			self.model.load_state_dict(torch.load(trainedModelPath))
		self.trainer = Trainer(model=self.model, lr=self.LR, gamma=self.GAMMA)
		
	def get_state(self):

		pixel_array = self.game.get_game_frame() 
		pixel_array = cv2.resize(pixel_array, (40,30), interpolation=cv2.INTER_CUBIC) # resizes the pixel array
		pixel_array = np.mean(pixel_array, axis=2) # converts the image to grayscale
		pixel_array = pixel_array/255.0 # Normalizes the values in the pixel array
		pixel_array = pixel_array.flatten() # Flattens the pixel array to a single axis

		head = self.game.snake.body[0].getPoint()
		food = self.game.food.getPoint()

		point_l = Point(head.x - self.BLOCK_SIZE, head.y)
		point_r = Point(head.x + self.BLOCK_SIZE, head.y)
		point_u = Point(head.x, head.y - self.BLOCK_SIZE)
		point_d = Point(head.x, head.y + self.BLOCK_SIZE)

		dir_l = self.game.snake.direction == Direction.LEFT
		dir_r = self.game.snake.direction == Direction.RIGHT
		dir_u = self.game.snake.direction == Direction.UP
		dir_d = self.game.snake.direction == Direction.DOWN

		game_info = [
			# Danger straight
			(dir_r and self.game.is_collision(point_r)) or
			(dir_l and self.game.is_collision(point_l)) or
			(dir_u and self.game.is_collision(point_u)) or
			(dir_d and self.game.is_collision(point_d)),
			
			# Danger right
			(dir_r and self.game.is_collision(point_d)) or
			(dir_l and self.game.is_collision(point_u)) or
			(dir_u and self.game.is_collision(point_r)) or
			(dir_d and self.game.is_collision(point_l)),

			# Danger left
			(dir_r and self.game.is_collision(point_u)) or
			(dir_l and self.game.is_collision(point_d)) or
			(dir_u and self.game.is_collision(point_l)) or
			(dir_d and self.game.is_collision(point_r)),

			# Move Direction
			dir_l,
			dir_r,
			dir_u,
			dir_d,

			# Food location
			food.x < head.x, # food left
			food.x > head.x, # food right
			food.y < head.y, # food up
			food.y > head.y  # food down
		]

		# state = np.concatenate((game_info, pixel_array), axis=None)
		state = game_info
		return np.array(state, dtype=float)

	def remember(self, state, action, reward, next_state, game_over):
		self.memory.append((state, action, reward, next_state, game_over)) # pop left if MAX_MEMORY reached

	def train_long_memory(self):
		if len(self.memory) > self.BATCH_SIZE:
			mini_sample = random.sample(self.memory, self.BATCH_SIZE) # list of tuples
		else:
			mini_sample = self.memory
		
		states, actions, rewards, next_states, game_overs = zip(*mini_sample)
		self.trainer.train_step(states, actions, rewards, next_states, game_overs)

	def train_short_memory(self, state, action, reward, next_state, game_over):
		self.trainer.train_step(state, action, reward, next_state, game_over)

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

	def optimize(self):
		plot_scores = []
		plot_mean_scores = []
		total_score = 0
		record = 0
		# Training loop
		while True:
			# get old state
			state_old = self.get_state()

			# get move
			final_move = self.select_action(state_old)

			# perform move and get new state
			reward, game_over, score = self.game.play_step(final_move)
			# game.save_game_frame()
			state_new = self.get_state()
			
			# Train short memory
			self.train_short_memory(state_old, final_move, reward, state_new, game_over)

			# remember
			self.remember(state_old, final_move, reward, state_new, game_over)

			if game_over:
				# train long memory
				self.game.reset()
				self.episodes += 1
				self.train_long_memory()

				if score > record:
					record = score
					self.model.save()
				print(f"Game: {self.episodes}, Score: {score}, Record: {record}")

				plot_scores.append(score)
				total_score += score
				mean_score = total_score / self.episodes
				plot_mean_scores.append(mean_score)
				plot(plot_scores, plot_mean_scores)