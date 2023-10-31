# UNC Charlotte
# ITCS 5153 - Applied AI - Fall 2023
# Lab 3
# Adversarial Search / Game Playing
# This program implements minimax and alpha beta pruning to
# be able to compete against user in a game of Connect 4

import numpy as np
import random
import pygame
import sys
import time
import math

# Constants for colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Board dimensions
ROW_COUNT = 6
COLUMN_COUNT = 7

# Players and pieces
PLAYER = 0
AI = 1
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

# Winning conditions
WINDOW_LENGTH = 4

# Initialize the game board with zeros
def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board

# Drop a piece into the specified column
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Check if a location in a column is a valid move
def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0

# Get the next open row in a column
def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

# Check if the board is full
def is_board_full(board):
    return all(board[ROW_COUNT - 1][col] != 0 for col in range(COLUMN_COUNT))

# Check if a player has a winning move on the board
def winning_move(board, piece):
    # Check horizontally
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True
    # Check vertically
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True
    # Check diagonally (from bottom-left to top-right)
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True
    # Check diagonally (from top-left to bottom-right)
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True

# Evaluate the score of a window for a specific player
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

# Score the position of the board for a specific player
def score_position(board, piece):
    score = 0

    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

# Check if the current board state is a terminal node
def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

# Minimax algorithm
def minimax(board, depth, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    start_time = time.time()
    nodes_explored = 0  # Initialize nodes_explored

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return None, 100000000000000, nodes_explored, 0
            elif winning_move(board, PLAYER_PIECE):
                return None, -10000000000000, nodes_explored, 0
            else:
                return None, 0, nodes_explored, 0
        else:
            return None, score_position(board, AI_PIECE), nodes_explored, 0

    if maximizingPlayer:
        best_value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            _, new_score, explored, _ = minimax(b_copy, depth - 1, False)
            nodes_explored += 1  # Increment nodes_explored for each explored node
            if new_score > best_value:
                best_value = new_score
                column = col
    else:
        best_value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            _, new_score, explored, _ = minimax(b_copy, depth - 1, True)
            nodes_explored += 1  # Increment nodes_explored for each explored node
            if new_score < best_value:
                best_value = new_score
                column = col

    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # in milliseconds
    return column, best_value, nodes_explored, elapsed_time

# Alpha-beta pruning version of the minimax algorithm
def alpha_beta_ab(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    start_time = time.time()
    nodes_explored = 0  # Initialize nodes_explored

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return None, 100000000000000, nodes_explored, 0
            elif winning_move(board, PLAYER_PIECE):
                return None, -10000000000000, nodes_explored, 0
            else:
                return None, 0, nodes_explored, 0
        else:
            return None, score_position(board, AI_PIECE), nodes_explored, 0

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            _, new_score, explored, _ = alpha_beta_ab(b_copy, depth - 1, alpha, beta, False)
            nodes_explored += 1  # Increment nodes_explored for each explored node
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            _, new_score, explored, _ = alpha_beta_ab(b_copy, depth - 1, alpha, beta, True)
            nodes_explored += 1  # Increment nodes_explored for each explored node
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break

    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # in milliseconds
    return column, value, nodes_explored, elapsed_time

# Get valid locations for a move on the current board
def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

# Choose the best move for the AI player
def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col

# Draw the game board
def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK,
                               (int(c * SQUARESIZE + SQUARESIZE / 2), int((r + 1) * SQUARESIZE + SQUARESIZE / 2)),
                               RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), height - int((r + 1) * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), height - int((r + 1) * SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    draw_button(minimax_radio_button, minimax_radio_button_color, "Minimax", (255, 255, 255))
    draw_button(alphabeta_radio_button, alphabeta_radio_button_color, "Alpha-Beta", (255, 255, 255))
    pygame.display.update()

# Create a button with the specified parameters
def create_button(x, y, width, height, color, text, text_color):
    button_rect = pygame.Rect(x, y + SQUARESIZE, width, height)
    pygame.draw.rect(screen, color, button_rect)
    pygame.draw.rect(screen, BLACK, button_rect, 5)

    font = pygame.font.SysFont("monospace", 30)
    label = font.render(text, 1, text_color)
    label_rect = label.get_rect(center=button_rect.center)
    screen.blit(label, label_rect)

    return button_rect

# Draw a button on the screen
def draw_button(button, color, text, text_color):
    pygame.draw.rect(screen, color, button)
    pygame.draw.rect(screen, BLACK, button, 5)

    font = pygame.font.SysFont("monospace", 30)
    label = font.render(text, 1, text_color)
    label_rect = label.get_rect(center=button.center)
    screen.blit(label, label_rect)

    return button, color  # Update the color variable

# Draw a button with hover effect
def draw_hover_button(button, hover_color, text, text_color):
    pygame.draw.rect(screen, hover_color, button)
    pygame.draw.rect(screen, BLACK, button, 5)

    font = pygame.font.SysFont("monospace", 30)
    label = font.render(text, 1, text_color)
    label_rect = label.get_rect(center=button.center)
    screen.blit(label, label_rect)

    return button, hover_color  # Update the color variable

# Initialize Pygame
pygame.init()

# Set up game parameters
SQUARESIZE = 100
width = (COLUMN_COUNT + 3) * SQUARESIZE
height = (ROW_COUNT + 2) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE / 2 - 5)

# Set up the game window
screen = pygame.display.set_mode(size)
myfont = pygame.font.SysFont("monospace", 75)

# Create buttons for new game, restart, and exit
new_game_button = create_button(COLUMN_COUNT * SQUARESIZE + 30, 50, 200, 50, (0, 255, 0), "New Game", (0, 0, 0))
restart_game_button = create_button(COLUMN_COUNT * SQUARESIZE + 30, 120, 200, 50, (255, 255, 0), "Restart", (0, 0, 0))
exit_button = create_button(COLUMN_COUNT * SQUARESIZE + 30, 190, 200, 50, (255, 0, 0), "Exit", (0, 0, 0))

# Create the game board
board = create_board()
game_over = False
turn = random.randint(PLAYER, AI)

# Create buttons for choosing the algorithm
minimax_radio_button = create_button(COLUMN_COUNT * SQUARESIZE + 30, 260, 200, 50, (255, 182, 193), "Minimax",
                                     (0, 0, 0))
alphabeta_radio_button = create_button(COLUMN_COUNT * SQUARESIZE + 30, 330, 200, 50, (173, 216, 230), "Alpha-Beta",
                                       (0, 0, 0))

# Set initial colors for algorithm buttons
minimax_radio_button_color = (255, 182, 193)
alphabeta_radio_button_color = (173, 216, 230)

# Set the current algorithm to minimax
current_algorithm = minimax

# Draw information on the surface
def draw_info(elapsed_time, nodes_explored):
    font = pygame.font.SysFont("monospace", 20)
    elapsed_time_text = font.render(f"Time: {elapsed_time:.3f} ms", True, (255, 255, 255))
    nodes_explored_text = font.render(f"Nodes Explored: {nodes_explored}", True, (255, 255, 255))
    screen.blit(elapsed_time_text, (width // 2 - 70, height - 70))
    screen.blit(nodes_explored_text, (width // 2 - 70, height - 40))


# Function to clear the banner
def clear_banner():
    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
    pygame.display.update()

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            # Handle mouse motion events for button hover effect
            if new_game_button.collidepoint(event.pos):
                draw_hover_button(new_game_button, (50, 205, 50), "New Game", (0, 0, 0))
            else:
                draw_button(new_game_button, (0, 255, 0), "New Game", (255, 255, 255))

            if restart_game_button.collidepoint(event.pos):
                draw_hover_button(restart_game_button, (255, 255, 0), "Restart", (0, 0, 0))
            else:
                draw_button(restart_game_button, (255, 255, 0), "Restart", (210, 210, 210))

            if exit_button.collidepoint(event.pos):
                draw_hover_button(exit_button, (255, 0, 0), "Exit", (0, 0, 0))
            else:
                draw_button(exit_button, (255, 0, 0), "Exit", (255, 255, 255))

            if minimax_radio_button.collidepoint(event.pos):
                draw_hover_button(minimax_radio_button, (255, 218, 185), "Minimax", (0, 0, 0))
            else:
                draw_button(minimax_radio_button, minimax_radio_button_color, "Minimax", (255, 255, 255))

            if alphabeta_radio_button.collidepoint(event.pos):
                draw_hover_button(alphabeta_radio_button, (135, 206, 250), "Alpha-Beta", (0, 0, 0))
            else:
                draw_button(alphabeta_radio_button, alphabeta_radio_button_color, "Alpha-Beta", (255, 255, 255))

        pygame.display.update()

        # Handle mouse click events
        if event.type == pygame.MOUSEBUTTONDOWN:
            try:
                # Start a new game
                if new_game_button.collidepoint(event.pos):
                    # Set the current algorithm based on the selected button
                    if minimax_radio_button_color == (255, 218, 185):
                        current_algorithm = minimax
                    elif alphabeta_radio_button_color == (135, 206, 250):
                        current_algorithm = alpha_beta_ab

                    # Initialize the game board and reset game parameters
                    board = create_board()
                    draw_board(board)
                    game_over = False  # Reset game_over flag
                    turn = random.randint(PLAYER, AI)

                    # Clear the winner text when the new game is initiated
                    screen.fill(BLACK, (0, 0, width, SQUARESIZE))

                # Restart the current game
                elif restart_game_button.collidepoint(event.pos):
                    # Initialize the game board and reset game parameters
                    board = create_board()
                    draw_board(board)
                    game_over = False  # Reset game_over flag
                    turn = random.randint(PLAYER, AI)

                    # Clear the winner text when the restart game is initiated
                    screen.fill(BLACK, (0, 0, width, SQUARESIZE))

                # Exit the game
                elif exit_button.collidepoint(event.pos):
                    sys.exit()

                # Select the Minimax algorithm
                elif minimax_radio_button.collidepoint(event.pos):
                    minimax_radio_button_color = (255, 140, 140)
                    alphabeta_radio_button_color = (173, 216, 230)

                # Select the Alpha-Beta Pruning algorithm
                elif alphabeta_radio_button.collidepoint(event.pos):
                    alphabeta_radio_button_color = (100, 160, 200)
                    minimax_radio_button_color = (255, 220, 230)

                # Handle player's move
                elif turn == PLAYER:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, PLAYER_PIECE)

                        if winning_move(board, PLAYER_PIECE):
                            label = myfont.render("Player 1 wins!!", 1, RED)
                            screen.blit(label, (40, 10))
                            game_over = True

                        turn += 1
                        turn %= 2

                        draw_board(board)
                        pygame.display.update()  # Update the screen to clear the message

            except IndexError:
                # Display a message when the player clicks out of bounds
                label = myfont.render("Play on to proceed", 1, (255, 255, 255))
                screen.blit(label, (40, 10))
                pygame.display.update()
                time.sleep(1)  # Pause for a short duration to make the message visible

    # AI's turn and other game logic
    if turn == AI and not game_over:
        clear_banner()
        # Set the current algorithm based on the selected button
        if minimax_radio_button_color == (255, 218, 185):
            current_algorithm = minimax
        elif alphabeta_radio_button_color == (135, 206, 250):
            current_algorithm = alpha_beta_ab

        # Get the AI's move using the chosen algorithm
        if current_algorithm == minimax:
            col, minimax_score, nodes_explored, elapsed_time = current_algorithm(board, 4, True)
        else:
            col, minimax_score, nodes_explored, elapsed_time = current_algorithm(board, 4, -math.inf, math.inf, True)

        # Make the AI's move
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)

            if winning_move(board, AI_PIECE):
                label = myfont.render("Player 2 wins!!", 1, YELLOW)
                screen.blit(label, (40, 10))
                game_over = True

            draw_board(board)

            turn += 1
            turn %= 2

            # Display information only for the AI's move
            pygame.draw.rect(screen, BLACK, (0, height - 70, width, 70))
            draw_info(elapsed_time, nodes_explored)

        # Check for a terminal state
        if is_terminal_node(board):
            if winning_move(board, PLAYER_PIECE):
                label = myfont.render("Player 1 wins!!", 1, RED)
                screen.blit(label, (40, 10))
            elif winning_move(board, AI_PIECE):
                label = myfont.render("Player 2 wins!!", 1, YELLOW)
                screen.blit(label, (40, 10))
            else:
                label = myfont.render("It's a draw!!", 1, RED)
                screen.blit(label, (40, 10))

            game_over = True

    pygame.display.update()
