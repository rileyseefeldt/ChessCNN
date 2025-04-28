import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.play_game import play_game
from src.play_game import ChessEngine

def main():

    path = "models/A/best.pth"

    engine = ChessEngine(model_path = path, num_simulations=1500, c_puct=1.5, temperature=0)

    play_game(engine)

if __name__ == "__main__":
    main()