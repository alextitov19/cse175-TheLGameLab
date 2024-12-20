from game_state import GameState
from utils import parse_move_input, validate_move, print_board, ntol
from minimax import alpha_beta_pruning
from heuristic import evaluate_board
import math
from visualizer import GameVisualizer  # Assuming GameVisualizer is defined


def human_move(game_state):
    """
    Prompt the human player to make a move.
    """
    print_board(game_state.board)
    while True:
        try:
            move_input = input("Enter your move (e.g., '1 2 E 4 3 1 1'): ")
            move = parse_move_input(move_input)
            if validate_move(game_state.board, move, game_state.current_player):
                return move
            else:
                print("Invalid move. Please try again.")
        except ValueError as e:
            print(f"Error: {e} Try again.")


def computer_move(game_state, depth=5):
    """
    Determine the computer's move using minimax with alpha-beta pruning.
    """
    print("Computer is thinking...")
    move, val = alpha_beta_pruning(
        state=game_state,
        depth=depth,
        alpha=-math.inf,
        beta=math.inf,
        maximizing_player=(game_state.current_player == 1),
        evaluate_fn=evaluate_board,
        get_legal_moves_fn=lambda state: state.get_legal_moves(state.current_player),
        apply_move_fn=lambda state, move: debug_apply_move(state, move),
        is_terminal_fn=lambda state: state.is_terminal(),
    )
    print("Computer's value:", val)
    adjusted_move = {
        "L_piece": {
            "x": move["L_piece"]["x"] + 1,
            "y": move["L_piece"]["y"] + 1,
            "config": ntol(move["L_piece"]["config"]),  # Configuration remains unchanged
        },
        "neutral_move": None if move["neutral_move"] is None else {
            "from": (move["neutral_move"]["from"][0] + 1, move["neutral_move"]["from"][1] + 1),
            "to": (move["neutral_move"]["to"][0] + 1, move["neutral_move"]["to"][1] + 1),
        },
    }

    s = f"{adjusted_move["L_piece"]["x"]} {adjusted_move["L_piece"]["y"]} {adjusted_move["L_piece"]["config"]}"

    if adjusted_move["neutral_move"]:
        s += f" {adjusted_move["neutral_move"]["from"][0]} {adjusted_move["neutral_move"]["from"][1]}"
        s += f" {adjusted_move["neutral_move"]["to"][0]} {adjusted_move["neutral_move"]["to"][1]}"

    print(f"Computer's move: {s}")

    # input("Press any key to continue...")  # Wait for key input
    return move

def main():
    """
    Main function to run the L-game.
    """
    print("\nWelcome to the L-game!\n")
    print("You can choose one of the following modes:")
    print("1. Human vs Human")
    print("2. Human vs Computer")
    print("3. Computer vs Computer\n")

    mode = input("Enter the mode number (1, 2, or 3): ")
    if mode not in {"1", "2", "3"}:
        print("Invalid mode. Exiting.")
        return
    
    x = 1
    if mode == "2":
        s = input("\nDo you want to go first? (y/n): ")
        if s == "n":
            x = 2
        elif s != "y":
            print("Invalid input. Exiting.")
            return

    game_state = GameState()
    visualizer = GameVisualizer(game_state.board)

    # Draw the initial board before the game loop starts
    visualizer.draw_board()

    while not game_state.is_terminal(check_ties=True):
        print(f"\nPlayer {game_state.current_player}'s turn.")

        if (mode == "1") or (mode == "2" and game_state.current_player == x):
            # Human move
            move = human_move(game_state)
            print("\nCurrent Board State:")
            print_board(game_state.board)
            game_state.apply_move(move)
            print("\nUpdated Board State:")
            print_board(game_state.board)
        

            # Update GUI immediately after the human move
            visualizer.update_board(game_state.board)
            visualizer.window.update()  # Force GUI to process all events
        else:
            # Computer move
            move = computer_move(game_state)
            game_state.apply_move(move)

            # Update GUI for the computer's move
            visualizer.update_board(game_state.board)

    # Final game over logic
    print("\nFinal Board State:")
    print_board(game_state.board)

    # Check tie conditions
    board_hash = game_state._hash_board()
    if game_state.state_history.get(board_hash, 0) >= game_state.max_repetitions:
        print("Game Over! It's a tie due to repeated states.")
    elif game_state.total_turns >= game_state.max_turns:
        print("Game Over! It's a tie due to exceeding the maximum number of turns.")
    else:
        print(f"Game Over! Player {3 - game_state.current_player} wins!")

    visualizer.run()  # Keeps the GUI window open at the end of the game
    
def debug_apply_move(state, move):
    """
    Debug wrapper for applying moves to a game state.
    """
    # print(f"Applying move: {move}")
    new_state = state.copy()
    if new_state is None:
        print("Error: state.copy() returned None.")
        return None
    try:
        new_state.apply_move(move)
        return new_state
    except Exception as e:
        print(f"Error during apply_move: {e}")
        return None
    
if __name__ == "__main__":
    main()