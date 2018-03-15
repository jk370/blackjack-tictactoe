import numpy as np
import blackjack

# Initialize V
v = {}

# Initialize policy
def policy(player_sum):
    '''Policy to stick if sum is 19 or higher, and stick otherwise'''
    if player_sum >= 19:
        return "stick"
    else:
        return "hit"
    

def initialise_V():
    '''Initialise all state-values for blackjack'''
    for player in range(12,22):
        for dealer in range(1,11):
            for ace in range(0,2):
                v[(player, dealer, ace)] = 0

def evaluate_policy():
    '''Evaluates the policy by playing episodes of blackjack (MCPE) and computes state values (v)'''
    # Initialise variable to hold all states with returns and times visited
    all_states_visited = {}
    total_episodes = 50000
    
    # Perform a high number of episodes
    for game in range(total_episodes):
        # Initialize game
        env = blackjack.Blackjack(verbose=False)
        new_state, reward = env.make_step(action="reset")
        end_state = np.array([-1, -1, -1])
        episode_visited_states = []

        # Play one episode
        while not np.array_equal(new_state, end_state):
            episode_visited_states.append((new_state[0], new_state[1], new_state[2]))
            player_sum = new_state[0]
            action = policy(player_sum)
            new_state, reward = env.make_step(action)

        # Store states and returns for episode
        for state in episode_visited_states:
            if state in all_states_visited: 
                old_reward, old_visits = all_states_visited[state]
                new_reward = old_reward + reward
                new_visits = old_visits + 1
                all_states_visited[state] = (new_reward, new_visits)
            else:
                all_states_visited[state] = (reward, 1)
        
    # Update V values
    for state, value in all_states_visited.items():
        total, visited = value
        average_return = total / visited
        v[state] = average_return
		
def get_state_value(s, v):
	''' Returns state value of given state'''
    value_of_s = v[s]
    return value_of_s
        

initialise_V()
evaluate_policy()
#for key, value in v.items():
    #print("State:", key, "\tValue: %0.3f" % value)