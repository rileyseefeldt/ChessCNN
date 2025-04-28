import time
import chess
import torch
import numpy as np
from src.model import ChessNet
from src.mcst import MCTSNode, expand_node, select_child, backpropagate
from src.preprocess import board_to_matrix

class ChessEngine:
    def __init__(self, model_path, num_simulations=800, c_puct=1.5, temperature=1):
        """Initialize the chess engine with a trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        #NEED TO CHANGE THIS SO IT WORKS
        self.model = ChessNet(num_filters=128, num_res_blocks=10)
        state_dict = torch.load(model_path)

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        self.model.load_state_dict(new_state_dict)
        self.model = torch.compile(self.model)
        self.model.to(self.device)
        
        # MCTS parameters
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
    
    def select_move(self, board):
        """Use MCTS to select the best move for the current position."""
        # Create root node
        root = MCTSNode(board.copy())
        
        # Expand the root node
        expand_node(root, self.model, self.device)
        root.visit_count = 1
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            # Selection phase
            node = root
            search_path = [node]
            
            while node.children and not node.state.is_game_over():
                node = select_child(node, self.c_puct)
                search_path.append(node)
            
            # Expansion phase (if needed)
            if node is not None and not node.children and not node.state.is_game_over():
                expand_node(node, self.model, self.device)
            
            # Evaluation phase
            if node is not None:
                # Get value from the model or terminal state
                if node.state.is_game_over():
                    # Terminal state value
                    if node.state.is_checkmate():
                        result = -1.0 if node.state.turn else 1.0  # Winner is the opposite of whose turn it is
                    else:
                        result = 0.0  # Draw
                else:
                    # Use the model to evaluate
                    board_tensor = board_to_matrix(node.state)
                    board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        _, value_pred = self.model(board_tensor)
                    
                    result = value_pred.item()
                
                # Backpropagation phase
                for node in reversed(search_path):
                    backpropagate(node, result)
                    result = -result  # Flip the result for the parent (opponent's perspective)
        
        # Calculate move probabilities based on visit counts
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        moves = list(root.children.keys())
        
        if self.temperature == 0.0:
            # Choose the move with highest visit count
            best_idx = np.argmax(visit_counts)
            selected_move = moves[best_idx]
        else:
            # Apply temperature and sample
            visit_counts = visit_counts ** (1 / self.temperature)
            probs = visit_counts / np.sum(visit_counts)
            selected_move = np.random.choice(moves, p=probs)
        
        print(f"Bot played {selected_move}")
        return selected_move