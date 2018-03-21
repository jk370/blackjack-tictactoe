from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class TicTacToe:
    def __init__(self, initial_state = np.full(9, " ", str)):
        '''Initialises game board and first player'''
        self.initial_state = deepcopy(initial_state)
        self.state = initial_state
        self.players = ["O", "X"]
        self.first_player = self.players[np.random.randint(0,2)]
    
    def get_initial_state(self):
        '''Returns initial state of the environment'''
        return deepcopy(self.initial_state)
    
    def get_state(self):
        '''Returns the current state representation'''
        return self.state
    
    def set_state(self, state):
        '''Sets the state of the game'''
        self.state = state
    
    def get_first_player(self):
        '''Returns the first player to take their turn'''
        return self.first_player
    
    def get_available_actions(self):
        '''Returns a list of available actions in the current state'''
        available_actions = []
        for idx, val in enumerate(self.state):
            if val == " ":
                available_actions.append(idx)
        return available_actions
    
    def print_board(self):
        '''Prints current state of the game board'''
        board = np.reshape(self.state, (3,3))
        for row in board:
            print("  %s  |  %s  |  %s  " % (row[0], row[1], row[2]))
        print("-----------------")
        print()
    
    def place_move(self, player, move):
        '''Places the player at the row and col provided'''
        if self.state[move] == " ":
            self.state[move] = player
        else:
            print("Not a valid move")
    
    def check_end(self, player):
        '''Checks if the episode is over'''
        end = self.check_lines(player) or self.check_diag(player) or self.check_draw(player)
        return end
    
    def check_lines(self, player):
        '''Checks rows and columns for a win'''
        board = np.reshape(self.state, (3,3))
        # Check each direction for a win
        for i in range(0,3):
            # Check rows
            if board[i][0] == board[i][1] and board[i][0] == board[i][2] and board[i][0] == player:
                return True
            
            # Check columns
            if board[0][i] == board[1][i] and board[0][i] == board[2][i] and board[0][i] == player:
                return True
        
        return False
    
    def check_diag(self, player):
        '''Checks diagonals for a win'''
        board = np.reshape(self.state, (3,3))    
        # Check diagonals
        if board[0][0] == board[1][1] and board[0][0] == board[2][2] and board[0][0] == player:
            return True
                
        elif board[2][0] == board[1][1] and board[2][0] == board[0][2] and board[2][0] == player:
            return True
            
        else:
            return False
        
    def check_draw(self, player):
        '''Checks if the game has been drawn'''
        win = self.check_lines(player) or self.check_diag(player)
        full = True
        for cell in self.state:
            if cell == " ":
                full = False
                break
                
        return (full and not win)
                
    def get_rewards(self, player):
        '''Returns a dictionary of reward for each player at the end of the episode'''
        episode_rewards = {
            "O": 0,
            "X": 0
        }
        # Amend rewards if not drawn (game has been won by player)
        if self.check_end(player) and not self.check_draw(player):
            episode_rewards[player] += 1
            if player == "O":
                episode_rewards["X"] -= 1
            else:
                episode_rewards["O"] -= 1
            
        return episode_rewards
    
    def reset_game(self):
        '''Resets the game board and randomises first player'''
        # Keep same board reference
        self.state = deepcopy(self.initial_state)
        self.first_player = self.players[np.random.randint(0,2)]