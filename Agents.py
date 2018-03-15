class Agent:
    def __init__(self, environment, player):
        '''Initialises environment and player'''
        self.environment = environment
        self.player = player
        self.cumulative_reward = [0]
        self.average_reward = [0]
        
    def set_environment(self, environment):
        '''Sets environment'''
        self.environment = environment
    
    def get_icon(self):
        '''Returns the player symbol'''
        return self.player
    
    def get_cumulative_reward(self):
        '''Returns cumulative reward array for player'''
        return self.cumulative_reward
    
    def get_average_reward(self):
        '''Returns average reward array for player'''
        return self.average_reward
    
    def add_cumulative_reward(self, reward):
        '''Appends cumulative reward to player cumulative reward history'''
        self.cumulative_reward.append(reward)
        
    def add_average_reward(self, reward):
        '''Appends average reward to player average reward history'''
        self.average_reward.append(reward)
        
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
		
class Q_Agent(Agent):
    def __init__(self, environment, player, epsilon = 0.1, alpha = 0.1, gamma = 1):
        ''' Initialises all learning variables and Q table'''
        # Set up initial variables and learning tables
        super(Q_Agent, self).__init__(environment, player)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
        
        # Knowledge of historic actions (due to turn-based game) - intialise first state-action
        self.last_state = self.environment.get_initial_state()
        self.last_action = 0
        key = (''.join(self.last_state), self.last_action)
        self.Q[key] = 0
    
    def get_last_turn(self):
        '''Returns the last state and action'''
        return (self.last_state, self.last_action)
    
    def set_last_turn(self, state, action):
        '''Saves the last turn in the memory'''
        self.last_state = state
        self.last_action = action
        
    def set_epsilon(self, epsilon):
        '''Sets the value of epsilon'''
        self.epsilon = epsilon
    
    def find_move(self):
        '''Chooses to exploit Q table, or explore'''
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
            # Initialise state and action values in Q table if not found
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
    
    def learn(self, old_state, action_t, new_state, reward):
        '''Updates Q table with value corresponding to latest state-action-reward'''
        available_actions = self.environment.get_available_actions()
        old_state = ''.join(old_state)
        new_state = ''.join(new_state)
        
        # Discover new state-actions
        for action in available_actions:
            key = (new_state, action)
            if key not in self.Q:
                self.Q[key] = 0
        
        # Check if terminal state
        if self.environment.check_end(self.player):
            factor = 0
        else:
            factor = max([self.Q[(new_state, action)] for action in available_actions])
        
        # Amend Q values
        self.Q[(old_state, action_t)] = ((1-self.alpha) * self.Q[(old_state, action_t)]) + (self.alpha * (reward + factor))