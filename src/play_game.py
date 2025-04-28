import os
import chess.pgn
import chess
import numpy as np
from src.engine_integration import ChessEngine

def get_player_move(board):
    while True:
        try:
            move_uci = input("\nEnter your move (in UCI format, e.g., 'e2e4'): ")
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move! Please try again.")
        except ValueError:
            print("Invalid format! Please use UCI format (e.g., 'e2e4')")

def save_game_to_file(game, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a', encoding='utf-8') as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)
        pgn_file.write("\n\n")

def play_game(engine, engine2 = None, opponent_type = 'human', player_color = None, starting_fen=None, save_data = False, max_moves = 200):
    if starting_fen:
        board = chess.Board(starting_fen)
    else:
        board = chess.Board()

    if player_color is None:
        player_color = chess.WHITE if np.random.rand() > 0.5 else chess.BLACK
    
    if save_data:
        game = chess.pgn.Game()
        game.setup(board)
        node = game
        game.headers["White"]  = "Player_1" if player_color == chess.WHITE else "Player_2"
        game.headers["Black"] = "Player_1" if player_color == chess.BLACK else "Player_2"


    for _ in range(max_moves):
        if board.is_game_over():
            break
        
        engine_turn = (
            (opponent_type == 'self') or 
            (opponent_type == 'human' and board.turn != player_color)
        )

        if engine_turn:
            if engine2 is None:
                move = engine.select_move(board)
            else:
                if board.turn == player_color:
                    move = engine.select_move(board)
                else:
                    move = engine2.select_move(board)
        else:
            move = get_player_move(board)

        board.push(move)
        if save_data:
            node = node.add_variation(move)

    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate():
        result = "1/2-1/2"
    elif board.is_insufficient_material():
        result = "1/2-1/2"
    elif board.is_fifty_moves():
        result = "1/2-1/2"
    elif board.is_repetition():
        result = "1/2-1/2"
    else:
        result = "*"

    print(f"Result: {result}")

    if(save_data):
        game.headers["Result"] = result
        save_game_to_file(game, "data/self_play.pgn")
    
    return result
    


    
