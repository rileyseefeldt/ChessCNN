# Turns PGN data into an HDF5 dataset for training a combined policy–value network

import chess.pgn
import h5py
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import board_to_matrix

PGN_FILE = "data/lichess_db_standard_rated_2025-01.pgn"
HDF5_FILE = "data/chess_data.h5"
TOTAL_GAMES = 100000
GAMES_PER_BATCH = 10000  # Process 100K games at a time
PROMO_OFFSET = {'q': 0, 'r': 1, 'b': 2, 'n': 3}

def move_to_index(move: chess.Move) -> int:
    """Maps a chess.Move to [0..4351]. Raises ValueError if unsupported."""
    from_sq, to_sq = move.from_square, move.to_square

    # Normal move: from*64 + to   → 0..4095
    if move.promotion is None:
        return from_sq * 64 + to_sq

    sym = chess.piece_symbol(move.promotion)
    if sym not in PROMO_OFFSET:
        raise ValueError(f"Unsupported promotion piece: {sym}")

    # Promotions packed by TO‑square: 4 slots each
    return 4096 + to_sq * 4 + PROMO_OFFSET[sym]

def process_pgn_to_hdf5(pgn_path, hdf5_path, total_games, games_per_batch):
    game_count = 0

    with open(pgn_path) as pgn_file, h5py.File(hdf5_path, "a") as hdf5_file:  # 'a' mode to append data
        # If datasets don't exist yet, create them
        if "X" not in hdf5_file:
            hdf5_file.create_dataset("X", shape=(0, 19, 8, 8), maxshape=(None, 19, 8, 8), dtype=np.float32)
            hdf5_file.create_dataset("Y", shape=(0, 2), maxshape=(None, 2), dtype=np.float32)

        while game_count < total_games:
            X, Y = [], []
            batch_count = 0

            while batch_count < games_per_batch and game_count < total_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                result = game.headers.get("Result")
                if result not in {"1-0", "0-1", "1/2-1/2"}:
                    continue

                # Set value target: 1 for white win, -1 for black win, 0 for draw
                value_label = 1 if result == "1-0" else -1 if result == "0-1" else 0

                board = game.board()
                moves = list(game.mainline_moves())

                for move in moves:
                    X.append(board_to_matrix(board))
                    policy_target = move_to_index(move)
                    adjusted_value = value_label if board.turn == chess.WHITE else -value_label
                    Y.append([policy_target, adjusted_value])
                    board.push(move)

                game_count += 1
                batch_count += 1
                if game_count % 1000 == 0:
                    print(f"Processed {game_count} games...")

            # Convert to NumPy arrays
            X_arr = np.array(X, dtype=np.float32)
            Y_arr = np.array(Y, dtype=np.float32)

            # Append to HDF5 file
            h5file_x = hdf5_file["X"]
            h5file_y = hdf5_file["Y"]

            h5file_x.resize((h5file_x.shape[0] + X_arr.shape[0]), axis=0)
            h5file_y.resize((h5file_y.shape[0] + Y_arr.shape[0]), axis=0)

            h5file_x[-X_arr.shape[0]:] = X_arr
            h5file_y[-Y_arr.shape[0]:] = Y_arr

            print(f"✅ Saved batch of {batch_count} games. Total saved: {game_count}/{total_games}")

if __name__ == "__main__":
    process_pgn_to_hdf5(PGN_FILE, HDF5_FILE, TOTAL_GAMES, GAMES_PER_BATCH)
