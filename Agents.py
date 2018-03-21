class Agent:
    def __init__(self, environment, player):
        '''Initializes environment and player'''
        self.environment = environment
        self.player = player
        self.cumulative_reward = [0]
        self.episode_reward = [0]
        
    def set_environment(self, environment):
        '''Sets environment'''
        self.environment = environment
    
    def get_icon(self):
        '''Returns the player symbol'''
        return self.player
    
    def get_cumulative_reward(self):
        '''Returns cumulative reward array for player'''
        return self.cumulative_reward
    
    def get_episode_reward(self):
        '''Returns average reward array for player'''
        return self.episode_reward
    
    def add_cumulative_reward(self, reward):
        '''Appends cumulative reward to player cumulative reward history'''
        self.cumulative_reward.append(reward)
        
    def add_episode_reward(self, reward):
        '''Appends average reward to player average reward history'''
        self.episode_reward.append(reward)
        
    def random_action(self):
        ''' Chooses a random action from those available'''
        available_actions = self.environment.get_available_actions()
        move = np.random.choice(available_actions)
        return move
		
		
class Random_Agent(Agent):
    def __init__(self, environment, player):
        '''Creates a reference to the game board to check for legal moves'''
        super(Random_Agent, self).__init__(environment, player)
        
    def find_move(self):
        '''Returns the first legal random move'''
        return self.random_action()
		
		
class Learning_Agent(Agent):
    def __init__(self, environment, player, epsilon = 0.1, alpha = 0.1, gamma = 1):
        ''' Initializes all learning variables and Q table'''
        # Set up initial variables and learning tables
        super(Learning_Agent, self).__init__(environment, player)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
        
    def get_Q(self):
        '''Returns Q table for the agent'''
        return self.Q
        
    def set_epsilon(self, epsilon):
        '''Sets the value of epsilon'''
        self.epsilon = epsilon
        
    def find_move(self):
        '''Chooses to exploit Q table, or explore (e-greedy)'''
        # Random action to explore, probability depends on epsilon
        chance = np.random.uniform(0,1)
        if (chance < self.epsilon):
            state_t = ''.join(self.environment.get_state())
            action = self.random_action()
            key = (state_t, action)
            # Add state-action to Q-dictionary if new
            if key not in self.Q:
                self.Q[key] = 0
            return action
        
        # Otherwise, try to be greedy
        else:
            # Initialize state and action values in Q table if not found
            state_t = ''.join(self.environment.get_state())
            available_actions = self.environment.get_available_actions()
            # Discover available states
            for action in available_actions:
                key = (state_t, action)
                if key not in self.Q:
                    self.Q[key] = 0

            # Find max Q from state      
            max_Q = max(self.Q[(state_t, action)] for action in available_actions)
            
            # Choose random action that corresponds to max_Q
            max_actions = []
            for action in available_actions:
                if (self.Q[(state_t, action)] == max_Q):
                    max_actions.append(action)
                    
            max_action = np.random.choice(max_actions)
            return max_action
			
			
class Q_Agent(Learning_Agent):
    def __init__(self, environment, player, epsilon = 0.1, alpha = 0.1, gamma = 1):
        ''' Initializes all learning variables and Q table'''
        super(Q_Agent, self).__init__(environment, player, epsilon, alpha, gamma)
    
    def learn(self, old_state, action_t, reward, new_state, new_action):
        '''Updates Q table with value corresponding to latest state-action-reward'''
        available_actions = self.environment.get_available_actions()
        old_state = ''.join(old_state)
        new_state = ''.join(new_state)
        update_key = (old_state, action_t)
        
        # Discover new state-actions
        for action in available_actions:
            key = (new_state, action)
            if key not in self.Q:
                self.Q[key] = 0
        
        # Check if terminal state
        if self.environment.check_end(self.player):
            factor = 0
        else:
            factor = self.gamma * max([self.Q[(new_state, action)] for action in available_actions])
        
        # Amend Q values
        self.Q[update_key] = self.Q[update_key] + (self.alpha * (reward + factor - self.Q[update_key]))
		
		
class Sarsa_Agent(Learning_Agent):
    def __init__(self, environment, player, epsilon = 0.1, alpha = 0.1, gamma = 1):
        ''' Initializes all learning variables and Q table'''
        super(Sarsa_Agent, self).__init__(environment, player, epsilon, alpha, gamma)
    
    def learn(self, old_state, action_t, reward, new_state, new_action):
        '''Updates Q table with value corresponding to latest state-action-reward'''
        old_state = ''.join(old_state)
        new_state = ''.join(new_state)
        key = (old_state, action_t)
        new_key = (new_state, new_action)
        
        # Discover new state if not previously found
        if new_key not in self.Q:
            self.Q[new_key] = 0
            
        # Amend Q values
        factor = (self.gamma * self.Q[new_key]) - self.Q[key]
        self.Q[key] = self.Q[key] + (self.alpha * (reward + factor))