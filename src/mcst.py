import numpy as np
import torch
import chess
from src.model import ChessNet
from src.preprocess import board_to_matrix
from scripts.generate_data import move_to_index

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

def puct_score(parent, child, c_puct: float) -> float:
    q_parent = -child.value()                                      # <-- flip!
    u_parent = (c_puct * child.prior *
                np.sqrt(parent.visit_count) / (1 + child.visit_count))
    return q_parent + u_parent

def select_child(node, c_puct: float):
    return max(node.children.values(),
               key=lambda ch: puct_score(node, ch, c_puct))

def expand_node(node, model, device):
    """Expand a node by adding one child per legal move,
       with priors taken from the policy network."""
    board = node.state
    policy_input = torch.tensor(
        board_to_matrix(board), dtype=torch.float32
    ).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        policy_logits, _ = model(policy_input)

    policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy().ravel()

    for move in board.legal_moves:
        idx = move_to_index(move)
        if idx is None or idx >= policy_probs.size:
            continue

        # --- create the child position without touching the parent ---
        child_board = board.copy(stack=False)
        child_board.push(move)                       # advance to child state

        child = MCTSNode(child_board, parent=node)
        child.prior = float(policy_probs[idx])
        node.children[move] = child

def backpropagate(node, result):
    """Propagate the value back up the tree"""
    while node is not None:
        node.visit_count += 1
        node.value_sum += result
        node = node.parent
