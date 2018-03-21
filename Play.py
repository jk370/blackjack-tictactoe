import TicTacToe
import Agents
from copy import deepcopy

def play(environment = TicTacToe(), policy = "Random", random_start = True, episodes = 10000):
    '''Creates two agents to play against each other'''
    # Initialize starting variables
    environment.reset_game()
    opponent = Random_Agent(environment = environment, player = "X")
    
    # Create agent with appropriate policy
    if policy == "Q":
        agent = Q_Agent(environment = environment, player = "O", epsilon = 0.1, alpha = 0.1, gamma = 1)
    elif policy == "Sarsa":
        agent = Sarsa_Agent(environment = environment, player = "O", epsilon = 0.1, alpha = 0.1, gamma = 1)
    else:
        agent = Random_Agent(environment = environment, player = "O")
    
    # Play episodes
    for episodes in range(episodes):
        # Reset episode to initial state
        environment.reset_game()
        game_end = False
        
        # Take first turn if opponent
        if random_start:
            first = environment.get_first_player()
            opponent_icon = opponent.get_icon()
            if first == opponent_icon:
                move = opponent.find_move()
                environment.place_move(opponent_icon, move)
        
        # Initialize agent move for Sarsa
        if type(agent) is Sarsa_Agent:
            agent_move = agent.find_move()
                
        # Plays entire episode
        while not game_end:
            # Make initial move with agent and observe reward
            state_t = deepcopy(environment.get_state())
            agent_icon = agent.get_icon()
            # Find move if not Sarsa agent
            if type(agent) is not Sarsa_Agent:
                agent_move = agent.find_move()
            environment.place_move(agent_icon, agent_move)
            rewards = environment.get_rewards(agent_icon)
            
            # Check if episode should continue
            if not environment.check_end(agent_icon):
                # Environment responds and observe new reward
                opponent_icon = opponent.get_icon()
                opponent_move = opponent.find_move()
                environment.place_move(opponent_icon, opponent_move)
                rewards = environment.get_rewards(opponent_icon)
            
            # Check if episode was ended by either player
            if environment.check_end(agent_icon) or environment.check_end(opponent_icon):
                action_t1 = "" # irrelevant as (state_t1, action_t1) equates to zero at terminal
                game_end = True
                
                # Add rewards to each player
                players = [agent, opponent]
                for player in players:
                    # Add episode rewards
                    last_reward = rewards[player.get_icon()]
                    player.add_episode_reward(last_reward)

                    # Add cumulative rewards
                    new_cumulative = player.get_cumulative_reward()[-1] + last_reward
                    player.add_cumulative_reward(new_cumulative)
            else:
                # Find next action for Sarsa
                action_t1 = agent.find_move()
                    
            # Learn from observations - only if learning agent
            if isinstance(agent, Learning_Agent):       
                reward = rewards[agent_icon]
                state_t1 = deepcopy(environment.get_state())
                agent.learn(state_t, agent_move, reward, state_t1, action_t1)
                agent_move = action_t1
                    
    # Return both random agents
    if policy == "Random":
        return players
    else:
        # Return agent after episodes have been played
        return agent

		
def plot_random_graph(players):
    '''Plots reward graph for episodes played'''
    plt.figure(1)
    icons = []
    for player in players:
        icons.append(player.get_icon())
        plt.plot(player.get_cumulative_reward())
    plt.title("Random Agents")
    plt.xlabel("Number of episodes")
    plt.ylabel("Cumulative Return")
    plt.legend(icons, loc = 'upper left')

	
def plot_Q_learning_graph(agents = 10, episode_number = 10000):
    '''Plots average reward graph for episodes played'''
    plt.figure(2)
    total_q_average = []
    
    # Collect returns of 10 Q-agents playing 10000 episodes each
    for agent in range(agents):
        q_agent = play(policy = "Q", episodes = episode_number)
        agent_rewards = q_agent.get_episode_reward()
        total_q_average.append(agent_rewards)
        
    # Average across 10 agents and 50 episodes (running average)
    total_q_average = np.mean(total_q_average, axis = 0, dtype = np.float64)
    total_q_average = np.convolve(total_q_average, np.ones((50,))/50, mode='valid')
    
    # Plot returns
    plt.plot(total_q_average)
    plt.title("Average Learning of 10 Q-Agents")
    plt.xlabel("Number of episodes")
    plt.ylabel("Average Return")
    
    # Return final list for future use
    return total_q_average

	
def find_optimal_play(state):
    # Complete 20000 episodes from given state with agent giving first move
    environment = TicTacToe(deepcopy(state))
    agent = play(environment = environment, policy = "Q", random_start = False, episodes = 20000)
    
    # Set environment back to given state and let learned agent take best move
    environment = TicTacToe(state)
    agent.set_environment(environment)
    agent.set_epsilon(0) # Ensures greedy move
    move = agent.find_move()
    
    # Uncomment to see Q-Values
    '''
    for action in environment.get_available_actions():
        state_val = ''.join(state)
        key = (state_val, action)
        print("Location:", action, "\tValue:", agent.get_Q()[key])
    '''
    
    # Place move and print board
    environment.place_move(agent.get_icon(), move)
    environment.print_board()

def plot_Sarsa_and_Q_learning_graphs(q_agent, agents = 10, episode_number = 10000):
    '''Plots average reward graph for episodes played'''
    plt.figure(3)
    total_sarsa_average = []
    
    # Collect returns of 10 Sarsa agents playing 10000 episodes
    for agent in range(agents):
        sarsa_agent = play(policy = "Sarsa", episodes = episode_number)
        agent_rewards = sarsa_agent.get_episode_reward()
        total_sarsa_average.append(agent_rewards)
    
    # Average across 10 agents and 50 episodes (running average)
    total_sarsa_average = np.mean(total_sarsa_average, axis = 0, dtype = np.float64)
    total_sarsa_average = np.convolve(total_sarsa_average, np.ones((50,))/50, mode='valid')
    
    # Plot returns
    plt.plot(q_agent)
    plt.plot(total_sarsa_average)
    plt.title("Average Learning of 10 agents each: Sarsa vs. Q-Learning")
    plt.xlabel("Number of episodes")
    plt.ylabel("Running Average Return")
    plt.legend(["Q", "Sarsa"], loc = "upper left")
	

# Play two random agents and plot reward graph
plot_random_graph(play(episodes = 10000))

# Average learning of 10 Q agents and plot graph
optimal_q_agent = plot_Q_learning_graph(agents = 10, episode_number = 10000)

# Test find optimal play function
state_test = ["O", "X", " ", " ", "X", " ", " ", "O", " "]
find_optimal_play(state_test)

# Compare Sarsa and Q-Learning Algorithms
plot_Sarsa_and_Q_learning_graphs(optimal_q_agent, agents = 10, episode_number = 10000)