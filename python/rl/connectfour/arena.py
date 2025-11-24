import torch
import numpy as np
from tqdm import tqdm # Optional: for a nice progress bar (pip install tqdm)
from model import QNConnectFour, ConnectFourEnv

# --- Ensure these classes are available (Copy them if in a new file) ---
# from your_script import ConnectFourEnv, QNConnectFour 

def load_model(path, input_dim=42, output_dim=7):
    """Helper to load a model architecture and weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNConnectFour(output_dim).to(device)
    try:
        # Load weights
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval() # Set to evaluation mode (turns off BatchNorm training)
        print(f"Successfully loaded: {path}")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
    return model

def get_model_move(model, board, available_moves, device):
    """
    Selects the best column based on Q-values.
    
    CRITICAL: The model always expects to view the board as 'Player 1'.
    If we are playing as Player 2 (-1), we must flip the board signs 
    before feeding it to the network.
    """
    # 1. Prepare tensor
    # Reshape 1D -> (1, 1, 6, 7)
    state_t = torch.tensor(board, dtype=torch.float32).unsqueeze(0).view(1, 1, 6, 7).to(device)
    
    # 2. Inference
    with torch.no_grad():
        q_values = model(state_t)
    
    # 3. Mask invalid moves
    # We set invalid moves to negative infinity
    mask = torch.full_like(q_values, -float('inf'))
    mask[0, available_moves] = q_values[0, available_moves]
    
    # 4. Argmax
    action = mask.max(1)[1].item()
    return action

def arena_battle(path_model_1, path_model_2, num_games=1000):
    env = ConnectFourEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n--- ⚔️ STARTING ARENA: {num_games} GAMES ⚔️ ---")
    print(f"Model 1: {path_model_1}")
    print(f"Model 2: {path_model_2}")
    
    # Load Models
    model1 = load_model(path_model_1)
    model2 = load_model(path_model_2)
    
    if model1 is None or model2 is None:
        return

    stats = {
        "Model_1_Wins": 0,
        "Model_2_Wins": 0,
        "Draws": 0
    }

    # Loop through games
    # We use tqdm for a progress bar, if you don't have it, just use range(num_games)
    iterator = tqdm(range(num_games)) if 'tqdm' in globals() else range(num_games)

    for i in iterator:
        # --- Manual Reset ---
        # We perform a manual reset to ensure NO random moves are made by the environment.
        env.board[:] = 0
        env.done = False
        env.winner = None
        
        # Randomize who starts to ensure fairness
        # 1 = Model 1, -1 = Model 2
        current_player = np.random.choice([1, -1]) 
        
        while not env.done:
            available_moves = env.available_actions_idx()
            
            if len(available_moves) == 0:
                stats["Draws"] += 1
                break

            if current_player == 1:
                # --- Model 1's Turn ---
                # Model 1 sees the board as is (1 is self, -1 is enemy)
                action = get_model_move(model1, env.board, available_moves, device)
                env.apply_action(action, 1)
                
                if env.check_win(1):
                    stats["Model_1_Wins"] += 1
                    break
            else:
                # --- Model 2's Turn ---
                # Model 2 sees board flipped (so it sees itself as 1)
                board_flipped = env.board * -1
                action = get_model_move(model2, board_flipped, available_moves, device)
                env.apply_action(action, -1)
                
                if env.check_win(-1):
                    stats["Model_2_Wins"] += 1
                    break
            
            # Switch turn
            current_player *= -1

    # --- Results ---
    print("\n\n=== 🏆 FINAL RESULTS ===")
    print(f"Model 1 Wins: {stats['Model_1_Wins']} ({(stats['Model_1_Wins']/num_games)*100:.1f}%)")
    print(f"Model 2 Wins: {stats['Model_2_Wins']} ({(stats['Model_2_Wins']/num_games)*100:.1f}%)")
    print(f"Draws:        {stats['Draws']} ({(stats['Draws']/num_games)*100:.1f}%)")
    print("========================\n")

# --- Example Usage ---
# Ensure you provide the correct paths to your .pth files
if __name__ == "__main__":    
    arena_battle("connect4_dqn_v1.pth", "connect4_dqn_v7.pth", num_games=1000)