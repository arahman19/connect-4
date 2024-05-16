# Connect 4 Game with Adversarial Search

This Python program implements a Connect 4 game where you can play against an AI opponent. The AI opponent uses two different algorithms for decision making: minimax with alpha-beta pruning.

## Game Rules

Connect 4 is a two-player connection game in which the players take turns dropping colored discs from the top into a grid. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs.

## Features

* Play against the computer AI.
* Choose between two AI algorithms: Minimax and Minimax with Alpha-Beta Pruning.
* Interactive GUI built with Pygame.

## Requirements

* Python 3.x
* Pygame library (pip install pygame)

## How to Play

1. Run the connect4.py script.
2. Choose the algorithm (Minimax or Alpha-Beta) by clicking on the corresponding button.
3. Click on the column where you want to drop your piece.
4. The AI will make its move, and the game continues until a player wins or the board is full.

## Implementation Details

* The game logic is implemented using a 2D array to represent the game board.
* The AI evaluates each possible move using the minimax algorithm, considering future states up to a specified depth.
* Alpha-beta pruning is applied to optimize the minimax algorithm and reduce the number of nodes explored.
* The GUI is built with Pygame, providing a visual representation of the game board and interactions with the player.
