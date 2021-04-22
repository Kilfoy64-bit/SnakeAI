try:
	import sys
	import pygame
	from pygame.locals import *	
	import random
	from enum import Enum
	from collections import namedtuple
	import numpy as np
except ImportError as err:
    print (f"couldn't load module. {err}")
    sys.exit(2)

pygame.init()

class Direction(Enum):
	RIGHT = 1
	LEFT = 2
	UP = 3
	DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGameAI:

	SCREEN_WIDTH = 640
	SCREEN_HEIGHT = 480

	BLACK = (0,0,0)
	WHITE = (255,255,255)
	RED = (255, 0, 0)
	BLUE = (0, 0, 255)
	GREEN = (0,128,0)
	LIGHT_GREEN = (0, 255, 0)
	BACKGROUND_COLOR = BLACK
	FONT_COLOR = WHITE
	
	BLOCK_SIZE = 20
	SPEED = 40
	
	font = pygame.font.SysFont('arial', 25)

	def __init__(self):
		# Initialise screen
		self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
		pygame.display.set_caption('Snake Game')
		self.clock = pygame.time.Clock()
		# Fill background
		background = pygame.Surface(self.screen.get_size())
		background = background.convert()
		background.fill(self.BACKGROUND_COLOR)
		# Initialise game state
		self.reset()
	
	def reset(self):
		# Initialise game state
		self.direction = Direction.RIGHT
		self.head = Point(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)
		self.snake = [self.head,
					  Point(self.head.x-self.BLOCK_SIZE, self.head.y),
					  Point(self.head.x-(2*self.BLOCK_SIZE), self.head.y)]
		self.score = 0
		self.food = None
		self._place_food()
		self.frame_iteration = 0

	def _place_food(self):
		x = random.randint(0, (self.SCREEN_WIDTH-self.BLOCK_SIZE)//self.BLOCK_SIZE) * self.BLOCK_SIZE
		y = random.randint(0, (self.SCREEN_HEIGHT-self.BLOCK_SIZE)//self.BLOCK_SIZE) * self.BLOCK_SIZE
		self.food = Point(x, y)
		if self.food in self.snake:
			self._place_food()
	
	def _update_ui(self):
		self.screen.fill(self.BLACK)

		for pt in self.snake:
			pygame.draw.rect(self.screen, self.GREEN, pygame.Rect(pt.x, pt.y, self.BLOCK_SIZE, self.BLOCK_SIZE))
			pygame.draw.rect(self.screen, self.LIGHT_GREEN, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
		
		pygame.draw.rect(self.screen, self.RED, pygame.Rect(self.food.x, self.food.y, self.BLOCK_SIZE, self.BLOCK_SIZE))

		text = self.font.render("Score: " + str(self.score), True, self.WHITE)
		self.screen.blit(text, [0, 0])
		pygame.display.flip()
	
	def _move(self, action):
		# [straight, right, left]

		clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
		index = clock_wise.index(self.direction)

		if np.array_equal(action, [1,0,0]):
			new_direction = clock_wise[index] # no change
		elif np.array_equal(action, [0,1,0]):
			next_index = (index + 1) % 4
			new_direction = clock_wise[next_index] # right turn
		else: # [0,0,1]
			next_index = (index - 1) % 4
			new_direction = clock_wise[next_index]
		
		self.direction = new_direction

		x = self.head.x
		y = self.head.y

		if self.direction == Direction.RIGHT:
			x += self.BLOCK_SIZE
		elif self.direction == Direction.LEFT:
			x -= self.BLOCK_SIZE
		elif self.direction == Direction.UP:
			y -= self.BLOCK_SIZE
		elif self.direction == Direction.DOWN:
			y += self.BLOCK_SIZE
		
		self.head = Point(x, y)
	
	def is_collision(self, pt=None):
		if pt is None:
			pt = self.head

		if pt.x > self.SCREEN_WIDTH - self.BLOCK_SIZE or pt.x < 0 or pt.y > self.SCREEN_HEIGHT - self.BLOCK_SIZE or pt.y < 0:
			return True
		
		if pt in self.snake[1:]:
			return True
		
		return False
	
	def play_step(self, action):
		self.frame_iteration += 1
		# 1. Collect user input
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				quit()
		# 2. Move
		self._move(action)
		self.snake.insert(0, self.head)
		# 3. Check if game over
		reward = 0
		game_over = False
		if self.is_collision() or self.frame_iteration > 100*len(self.snake):
			game_over = True
			reward = -10
			return reward, game_over, self.score
		# 4. Place new food or just move
		if self.head == self.food:
			self.score += 1
			reward = 10
			self._place_food()
		else:
			self.snake.pop()
		# 5. Update ui and clock
		self._update_ui()
		self.clock.tick(self.SPEED)
		# 6 Return game over and score
		
		return reward, game_over, self.score