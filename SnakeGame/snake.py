try:
	import settings
	import objects
	import sys
except ImportError as err:
    print (f"couldn't load module. {err}")
    sys.exit(2)

def main():
	game_settings = settings.Settings()
	game = objects.SnakeGame(game_settings)

	# Event loop
	while True:
		game_over, score = game.play_step()

		if game_over == True:
			print(f"Final Score: {score}")
			break

if __name__ == '__main__': main()