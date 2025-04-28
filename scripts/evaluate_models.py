#!/usr/bin/env python3
"""
Simple evaluation script for **one** ChessNet model versus Stockfish.

Edit the CONSTANTS section, then run:

    python evaluate_models.py

It will print PASS/FAIL for each FEN and a final summary.
"""

import json
import torch
import chess
import chess.engine
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.model import ChessNet
from src.preprocess import board_to_matrix

# ─────────── CONSTANTS ────────────
MODEL_PATH      = "models/E/best.pth"     # model to test
FEN_FILE        = "fens.json"             # JSON file: [{"depth":…, "nodes":…, "fen":"…"}, …]
STOCKFISH_PATH  = "/mnt/c/Users/riley/stockfish/stockfish.exe"  # path to Stockfish binary
STOCKFISH_DEPTH = 12                      # analysis depth (OK to raise/lower)
NUM_FILTERS    = 192                    # Filters for model variation
NUM_RESBLOCKS  = 20                       # ResBlocks for model variation
# ───────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChessNet(num_filters=NUM_FILTERS, num_res_blocks=NUM_RESBLOCKS).to(device)
raw_state = torch.load(MODEL_PATH, map_location=device)
clean_state = { (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in raw_state.items() }
model.load_state_dict(clean_state, strict=True)
model.eval()

# 2. Load FEN list -----------------------------------------------------------
with open(FEN_FILE) as f:
    fens_data = json.load(f)

# extract just the FEN strings
try:
    fens = [entry["fen"] for entry in fens_data]
except (TypeError, KeyError):
    # fallback if it's already a simple list of strings
    fens = list(fens_data)

print(f"Loaded {len(fens)} FEN positions from {FEN_FILE}")

# 3. Start Stockfish --------------------------------------------------------
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
print(f"Using Stockfish at {STOCKFISH_PATH}, depth {STOCKFISH_DEPTH}")

# Helper: convert Stockfish score
def sf_score_to_value(score: chess.engine.PovScore) -> float:
    """Convert Stockfish score to ChessNet value in [-1,1]."""
    if score.is_mate():
        return 1.0 if score.white().mate() > 0 else -1.0
    cp = score.white().score()
    return max(-1.0, min(1.0, cp / 1000.0))

# 4. Evaluate ---------------------------------------------------------------
total_diff = 0.0

for idx, fen in enumerate(fens, 1):
    board = chess.Board(fen)

    # — model value —
    x = torch.tensor(board_to_matrix(board), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, value_pred = model(x)
    model_val = value_pred.item()

    if not board.turn:
        model_val = -model_val

    # — Stockfish value —
    info = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
    sf_val = sf_score_to_value(info["score"])

    # — Difference —
    diff = abs(model_val - sf_val)
    total_diff += diff

    print(f"{idx:2d}/{len(fens)}  model={model_val:+.3f}  stockfish={sf_val:+.3f}  diff={diff:+.3f}")

# 5. Summary ---------------------------------------------------------------
average_diff = total_diff / len(fens)
print(f"\nSummary: Average difference across {len(fens)} positions = {average_diff:.4f}")

engine.quit()
