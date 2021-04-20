try:
	import sys
	import pygame
	from pygame.locals import *	
	import random
	from enum import Enum
	from collections import namedtuple
except ImportError as err:
    print (f"couldn't load module. {err}")
    sys.exit(2)

class Direction(Enum):
	RIGHT = 1
	LEFT = 2
	UP = 3
	DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGame:
	pygame.init()
	font = pygame.font.SysFont('arial', 25)

	def __init__(self, game_settings):
		self.game_settings = game_settings
		# Initialise screen
		self.screen = pygame.display.set_mode((game_settings.SCREEN_WIDTH, game_settings.SCREEN_HEIGHT))
		pygame.display.set_caption('Snake Game')
		self.clock = pygame.time.Clock()
		# Fill background
		background = pygame.Surface(self.screen.get_size())
		background = background.convert()
		background.fill(game_settings.BACKGROUND_COLOR)

		# Initialise game state
		self.direction = Direction.RIGHT

		self.head = Point(self.game_settings.SCREEN_WIDTH/2, self.game_settings.SCREEN_HEIGHT/2)
		self.snake = [self.head,
					  Point(self.head.x-game_settings.BLOCK_SIZE, self.head.y),
					  Point(self.head.x-(2*game_settings.BLOCK_SIZE), self.head.y)]
		
		self.score = 0
		self.food = None
		self._place_food()
	
	def _place_food(self):
		x = random.randint(0, (self.game_settings.SCREEN_WIDTH-self.game_settings.BLOCK_SIZE)//self.game_settings.BLOCK_SIZE) * self.game_settings.BLOCK_SIZE
		y = random.randint(0, (self.game_settings.SCREEN_HEIGHT-self.game_settings.BLOCK_SIZE)//self.game_settings.BLOCK_SIZE) * self.game_settings.BLOCK_SIZE
		self.food = Point(x, y)
		if self.food in self.snake:
			self._place_food()
	
	def _update_ui(self):
		self.screen.fill(self.game_settings.BLACK)

		for pt in self.snake:
			pygame.draw.rect(self.screen, self.game_settings.GREEN, pygame.Rect(pt.x, pt.y, self.game_settings.BLOCK_SIZE, self.game_settings.BLOCK_SIZE))
			pygame.draw.rect(self.screen, self.game_settings.LIGHT_GREEN, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
		
		pygame.draw.rect(self.screen, self.game_settings.RED, pygame.Rect(self.food.x, self.food.y, self.game_settings.BLOCK_SIZE, self.game_settings.BLOCK_SIZE))

		text = self.font.render("Score: " + str(self.score), True, self.game_settings.WHITE)
		self.screen.blit(text, [0, 0])
		pygame.display.flip()
	
	def _move(self, direction):
		x = self.head.x
		y = self.head.y

		if direction == Direction.RIGHT:
			x += self.game_settings.BLOCK_SIZE
		elif direction == Direction.LEFT:
			x -= self.game_settings.BLOCK_SIZE
		elif direction == Direction.UP:
			y -= self.game_settings.BLOCK_SIZE
		elif direction == Direction.DOWN:
			y += self.game_settings.BLOCK_SIZE
		
		self.head = Point(x, y)
	
	def _is_collision(self):
		if self.head.x > self.game_settings.SCREEN_WIDTH - self.game_settings.BLOCK_SIZE or self.head.x < 0 or self.head.y > self.game_settings.SCREEN_HEIGHT - self.game_settings.BLOCK_SIZE or self.head.y < 0:
			return True
		
		if self.head in self.snake[1:]:
			return True
		
		return False
	
	def play_step(self):
		# 1. Collect user input
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				quit()
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_LEFT:
					self.direction = Direction.LEFT
				elif event.key == pygame.K_RIGHT:
					self.direction = Direction.RIGHT
				elif event.key == pygame.K_UP:
					self.direction = Direction.UP
				elif event.key == pygame.K_DOWN:
					self.direction = Direction.DOWN
			
		# 2. Move
		self._move(self.direction)
		self.snake.insert(0, self.head)
		# 3. Check if game over
		game_over = False
		if self._is_collision():
			game_over = True
			return game_over, self.score
		# 4. Place new food or just move
		if self.head == self.food:
			self.score += 1
			self._place_food()
		else:
			self.snake.pop()
		# 5. Update ui and clock
		self._update_ui()
		self.clock.tick(self.game_settings.SPEED)
		# 6 Return game over and score
		
		return game_over, self.score