try:
	import sys
	import random
	from enum import Enum
	from collections import namedtuple

	import numpy as np
	import pygame
	from pygame.locals import *	
	
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
	def __init__(self, point: Point, size: int):
		self._point = point
		self._size = size

	def draw(self, innerColor, outerColor, surface):
		pygame.draw.rect(surface, innerColor, pygame.Rect(self._point.x, self._point.y, self._size, self._size))
		pygame.draw.rect(surface, outerColor, pygame.Rect(self._point.x+4, self._point.y+4, 12, 12))

	def getPoint(self):
		return self._point
	
	def setPoint(self, point: Point):
		self._point = point

class Snake:
	# HEAD COLORS
	LIGHT_PINK = (255, 0, 255)
	PINK = (128, 0, 128)
	# BODY COLORS
	GREEN = (0,128,0)
	LIGHT_GREEN = (0, 255, 0)

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

	def draw(self, surface):
		self.body[0].draw(self.PINK, self.LIGHT_PINK, surface)
		for block in self.body[1:]:
			block.draw(self.GREEN, self.LIGHT_GREEN, surface)


class SnakeGame:

	SURFACE_WIDTH = 640
	SURFACE_HEIGHT = 480

	BLACK = (0,0,0)
	WHITE = (255,255,255)
	LIGHT_RED = (255, 0, 0)
	RED = (128, 0, 0)

	BACKGROUND_COLOR = BLACK
	FONT_COLOR = WHITE
	
	BLOCK_SIZE = 20
	SPEED = 20
	
	font = pygame.font.SysFont('arial', 25)

	def __init__(self):
		# Initialise surface
		self.surface = pygame.display.set_mode((self.SURFACE_WIDTH, self.SURFACE_HEIGHT))
		pygame.display.set_caption('Snake Game')
		self.clock = pygame.time.Clock()
		# Fill background
		background = pygame.Surface(self.surface.get_size())
		background = background.convert()
		background.fill(self.BACKGROUND_COLOR)
		# Initialise game state
		self.reset()
	
	def reset(self):
		# Initialise game state
		self.snake = Snake(self.SURFACE_WIDTH/2, self.SURFACE_HEIGHT/2, self.BLOCK_SIZE)
		self.score = 0
		self.food = None
		self._place_food()
		self.frame_iteration = 0

	def _place_food(self):
		x = random.randint(0, (self.SURFACE_WIDTH-self.BLOCK_SIZE)//self.BLOCK_SIZE) * self.BLOCK_SIZE
		y = random.randint(0, (self.SURFACE_HEIGHT-self.BLOCK_SIZE)//self.BLOCK_SIZE) * self.BLOCK_SIZE
		self.food = Block(Point(x, y), self.BLOCK_SIZE)

		for block in self.snake.body:
			if self.food.getPoint() == block.getPoint():
				self._place_food()
	
	def _update_ui(self):
		self.surface.fill(self.BACKGROUND_COLOR)

		self.snake.draw(self.surface)
		
		self.food.draw(self.RED, self.LIGHT_RED, self.surface)

		text = self.font.render("Score: " + str(self.score), True, self.WHITE)
		self.surface.blit(text, [0, 0])
		pygame.display.flip()
	
	def _convert_userinput(self, direction):
		clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
		index = clock_wise.index(self.snake.direction)
		if direction == clock_wise[index]:
			return [1,0,0] # No change, move straight
		elif direction == clock_wise[(index - 2) % 4]:
			return [1,0,0] # No change, Attempted reverse illegal move
		elif direction == clock_wise[(index - 1) % 4]:
			return [0,0,1] # Right turn
		elif direction == clock_wise[(index + 1) % 4]:
			return [0,1,0] # Left turn
		
	
	def is_collision(self, point=None):
		if point is None:
			point = self.snake.head.getPoint()

		if point.x > self.SURFACE_WIDTH - self.BLOCK_SIZE or point.x < 0 or point.y > self.SURFACE_HEIGHT - self.BLOCK_SIZE or point.y < 0:
			return True
		
		for block in self.snake.body[1:]:	
			if point == block.getPoint():
				return True
		
		return False

	def play_step(self):
		action = [1, 0, 0] # No change in direction

		# User input
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				quit()
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_LEFT:
					action = self._convert_userinput(Direction.LEFT)
				elif event.key == pygame.K_RIGHT:
					action = self._convert_userinput(Direction.RIGHT)
				elif event.key == pygame.K_UP:
					action = self._convert_userinput(Direction.UP)
				elif event.key == pygame.K_DOWN:
					action = self._convert_userinput(Direction.DOWN)

		self.snake.move(action)

		game_over = False
		if self.is_collision():
			game_over = True
			return game_over, self.score

		if self.snake.head.getPoint() == self.food.getPoint():
			self.score += 1
			self._place_food()
		else:
			self.snake.body.pop()

		self._update_ui()
		self.clock.tick(self.SPEED)

		return game_over, self.score

class SnakeGameAI(SnakeGame):

	SPEED = 40

	def __init__(self):
		super().__init__()
	
	def get_game_frame(self):
		return pygame.surfarray.array3d(self.surface)
		
	def play_step(self, action):
		self.frame_iteration += 1

		# User input
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				quit()

		self.snake.move(action)

		reward = 0
		game_over = False
		if self.is_collision() or self.frame_iteration > 100*len(self.snake.body):
			game_over = True
			reward = -10
			return reward, game_over, self.score

		if self.snake.head.getPoint() == self.food.getPoint():
			self.score += 1
			reward = 10
			self._place_food()
		else:
			self.snake.body.pop()

		self._update_ui()
		self.clock.tick(self.SPEED)
		
		return reward, game_over, self.score