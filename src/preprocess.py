# Converts a chess.Board into a tensor with 19 channels
import numpy as np
import chess

PIECE_ENCODING = {           # unchanged
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12
}

TOTAL_CHANNELS = 19          # 14 original + side + 4 castling = 19

def board_to_matrix(board: chess.Board) -> np.ndarray:
    """
    Channels
    --------
    0‑11 : piece encodings (white then black)
    12   : legal‑move target squares
    13   : from‑squares of legal moves
    14   : side‑to‑move   (all‑ones for White, all‑zeros for Black)
    15   : White K‑side   castling rights
    16   : White Q‑side   castling rights
    17   : Black K‑side   castling rights
    18   : Black Q‑side   castling rights
    The board is always oriented so that *side to move* is at the bottom.
    """
    #rotates board if blacks turn
    flip = board.turn == chess.BLACK
    mat  = np.zeros((TOTAL_CHANNELS, 8, 8), dtype=np.float32)

    #pieces
    for sq, piece in board.piece_map().items():
        r, c = divmod(sq, 8)
        if flip:
            r, c = 7 - r, 7 - c
        mat[PIECE_ENCODING[piece.symbol()] - 1, r, c] = 1

    #legal moves
    #from squares and to squares
    for mv in board.legal_moves:
        to_r, to_c = divmod(mv.to_square, 8)
        if flip:
            to_r, to_c = 7 - to_r, 7 - to_c
        mat[12, to_r, to_c] = 1

    from_squares = {mv.from_square for mv in board.legal_moves}
    for sq in from_squares:
        fr, fc = divmod(sq, 8)
        if flip:
            fr, fc = 7 - fr, 7 - fc
        mat[13, fr, fc] = 1

    #side to move channel
    if board.turn == chess.WHITE:
        mat[14, :, :] = 1.0        # all ones for white, zeros for black

    #castling rights
    if board.has_kingside_castling_rights(chess.WHITE):  mat[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE): mat[16, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):  mat[17, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK): mat[18, :, :] = 1

    return mat
