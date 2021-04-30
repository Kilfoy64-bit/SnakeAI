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

class Block:
	
	GREEN = (0,128,0)
	LIGHT_GREEN = (0, 255, 0)

	def __init__(self, point: Point, size: int):
		self._point = point
		self._size = size

	def draw(self, innerColor, outerColor, screen):
		pygame.draw.rect(screen, innerColor, pygame.Rect(self._point.x, self._point.y, self._size, self._size))
		pygame.draw.rect(screen, outerColor, pygame.Rect(self._point.x+4, self._point.y+4, 12, 12))

	def getPoint(self):
		return self._point
	
	def setPoint(self, point: Point):
		self._point = point

class Snake:
	def __init__(self, xPos, yPos, blockSize):
		
		self.direction = Direction.RIGHT
		self.block_size = blockSize

		self.head = Block(Point(xPos, yPos), self.block_size)
		
		self.body = [self.head,
					 Block(Point(xPos - self.block_size, yPos), self.block_size),
					 Block(Point(xPos - (2*self.block_size), yPos), self.block_size)]
	
	def move(self, action):
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

		head = self.head.getPoint()
		x = head.x
		y = head.y

		if self.direction == Direction.RIGHT:
			x += self.block_size
		elif self.direction == Direction.LEFT:
			x -= self.block_size
		elif self.direction == Direction.UP:
			y -= self.block_size
		elif self.direction == Direction.DOWN:
			y += self.block_size
		
		self.head = Block(Point(x, y), self.block_size)
		self.body.insert(0, self.head)

	def draw(self, innerColor, outerColor, screen):
		for block in self.body:
			block.draw(innerColor, outerColor, screen)

class SnakeGameAI:

	SCREEN_WIDTH = 640
	SCREEN_HEIGHT = 480

	BLACK = (0,0,0)
	WHITE = (255,255,255)
	LIGHT_RED = (255, 0, 0)
	RED = (128, 0, 0)
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
		self.snake = Snake(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2, self.BLOCK_SIZE)
		self.score = 0
		self.food = None
		self._place_food()
		self.frame_iteration = 0

	def _place_food(self):
		x = random.randint(0, (self.SCREEN_WIDTH-self.BLOCK_SIZE)//self.BLOCK_SIZE) * self.BLOCK_SIZE
		y = random.randint(0, (self.SCREEN_HEIGHT-self.BLOCK_SIZE)//self.BLOCK_SIZE) * self.BLOCK_SIZE
		self.food = Block(Point(x, y), self.BLOCK_SIZE)

		for block in self.snake.body:
			if self.food.getPoint() in block.getPoint():
				self._place_food()
	
	def _update_ui(self):
		self.screen.fill(self.BLACK)

		self.snake.draw(self.GREEN, self.LIGHT_GREEN, self.screen)
		
		self.food.draw(self.RED, self.LIGHT_RED, self.screen)

		text = self.font.render("Score: " + str(self.score), True, self.WHITE)
		self.screen.blit(text, [0, 0])
		pygame.display.flip()
	
	def is_collision(self, point=None):
		if point is None:
			point = self.snake.head.getPoint()

		if point.x > self.SCREEN_WIDTH - self.BLOCK_SIZE or point.x < 0 or point.y > self.SCREEN_HEIGHT - self.BLOCK_SIZE or point.y < 0:
			return True
		
		for block in self.snake.body[1:]:	
			if point in block.getPoint():
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
		self.snake.move(action)
		# 3. Check if game over
		reward = 0
		game_over = False
		if self.is_collision() or self.frame_iteration > 100*len(self.snake.body):
			game_over = True
			reward = -10
			return reward, game_over, self.score
		# 4. Place new food or just move
		if self.snake.head.getPoint() == self.food.getPoint():
			self.score += 1
			reward = 10
			self._place_food()
		else:
			self.snake.body.pop()
		# 5. Update ui and clock
		self._update_ui()
		self.clock.tick(self.SPEED)
		# 6 Return game over and score
		
		return reward, game_over, self.score