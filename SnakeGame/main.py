try:
	import sys
	import objects
	import agent
except ImportError as err:
    print (f"couldn't load module. {err}")
    sys.exit(2)

def playSnakeGame():

	game = objects.SnakeGame()

	while True:

		game_over, score = game.play_step()

		if game_over == True:
			print(f"Final Score: {score}")
			break


def trainSnakeGame():
	agent.train()

def playSnakeGameAI(filepath):
	# Use learned paramaters to play Snake Game AI.
	pass

def main():
	playSnakeGame()

if __name__ == "__main__": main()