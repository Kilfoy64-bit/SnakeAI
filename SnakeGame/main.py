try:
	import sys
	import objects
	from agent import Agent
except ImportError as err:
    print (f"couldn't load module. {err}")
    sys.exit(2)

def playSnakeGame(): # Human input to play Snake
	game = objects.SnakeGame()
	while True:
		game_over, score = game.play_step()
		if game_over == True:
			print(f"Final Score: {score}")
			break

def trainSnakeGame(): # Train agent from scratch
	agent = Agent()
	agent.train()

def playSnakeGameAI(filepath): # Train agent from previously trained model
	agent = Agent(filepath)
	agent.train()

def main():
	# playSnakeGame()
	trainSnakeGame()
	# playSnakeGameAI("./model/model-V004.pth")

if __name__ == "__main__": main()