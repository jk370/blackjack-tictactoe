import TicTacToe
import Agents
from copy import deepcopy

def play(environment = TicTacToe(), policy = "Random", random_start = True, games = 10000):
    '''Creates two random agents to play against each other'''
    # Initialise starting variables
    opponent = Random_Agent(environment = environment, player = "X")
    
    # Create agent with appropriate policy
    if policy == "Random":
        agent = Random_Agent(environment = environment, player = "O")
    elif policy == "Q":
        agent = Q_Agent(environment = environment, player = "O", epsilon = 0.1, alpha = 0.1, gamma = 1)
    
    # Play games
    for game in range(games):
        # Reset game to initial state
        environment.reset_game()
        game_end = False
        players = [agent, opponent]
        
        # Randomly assign starting player if required
        if random_start:
            first = environment.get_first_player()
            if first == "X":
                players = [opponent, agent]

        # Plays entire game
        while not game_end:
            # Make a move for each player
            for player in players:
                # Store all required variables
                state_t = deepcopy(environment.get_state())
                icon = player.get_icon()
                move = player.find_move()
                environment.place_move(icon, move)
                state_t1 = deepcopy(environment.get_state())
                rewards = environment.get_rewards(icon)
                
                # Allow the Q-Agent to learn from each turn (if applicable)
                if type(player) is Q_Agent:
                    player.set_last_turn(state_t, move)
                    reward = rewards[icon]
                    player.learn(state_t, move, state_t1, reward)
                elif type(agent) is Q_Agent:
                    last_state, last_action = agent.get_last_turn()
                    reward = rewards[agent.get_icon()]
                    agent.learn(last_state, last_action, state_t1, reward)

                # Check if game has ended
                if environment.check_end(icon):
                    # Append cumulative and average rewards and end
                    for player in players:
                        last_reward = rewards[player.get_icon()]
                        # Average rewards
                        games_played = game + 1
                        current_average = player.get_average_reward()[-1]
                        new_average = ((games_played-1)*current_average+last_reward)/games_played
                        player.add_average_reward(new_average)
                        
                        # Cumulative rewards
                        new_cumulative = player.get_cumulative_reward()[-1] + last_reward
                        player.add_cumulative_reward(new_cumulative)
                        
                    game_end = True
                    break
                    
    # Plot graph for random agent
    if policy == "Random":
        plot_reward_graph(players)
    else:
        # Return agent after games have been played
        return agent
            
def plot_reward_graph(players):
    '''Plots reward graph for games played'''
    plt.figure(1)
    icons = []
    for player in players:
        icons.append(player.get_icon())
        plt.plot(player.get_cumulative_reward())
    plt.title("Random Agents")
    plt.xlabel("Number of games")
    plt.ylabel("Cumulative Return")
    plt.legend(icons, loc = 'upper left')
	
def plot_learning_graph(agents = 10):
    '''Plots average reward graph for games played'''
    plt.figure(2)
    total_average = []
    
    # Average the returns of 10 agents playing 2000 games
    for agent in range(agents):
        average_returns = play(policy = "Q", games = 2000).get_average_reward()
        total_average.append(average_returns)
    total_average = np.mean(total_average, axis = 0)
    
    # Plot returns
    plt.plot(total_average)
    plt.title("Average Learning of 10 Q-Agents")
    plt.xlabel("Number of games")
    plt.ylabel("Average Return")

def find_optimal_play(state):
    # Complete 10000 games from given state with agent giving first move
    environment = TicTacToe(deepcopy(state))
    agent = play(environment = environment, policy = "Q", random_start = False, games = 10000)
    
    # Set environment back to given state and let learned agent take best move
    environment = TicTacToe(state)
    agent.set_environment(environment)
    agent.set_epsilon(0) # Ensures greedy move
    move = agent.find_move()
    
    # Place move and print board
    environment.place_move(agent.get_icon(), move)
    environment.print_board()
    
# Play two random agents and plot reward graph
play(games = 10000)

# Average learning of 10 Q agents and plot graph
plot_learning_graph(agents = 10)

# Test find optimal play function
state_test = ["O", "X", " ", " ", "X", " ", " ", "O", " "]
find_optimal_play(state_test)