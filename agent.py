import random
import pprint 
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        # Initialize variables here
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.state = ()  # Initialize state
        self.previous_state = ()  # Initialize previous_state
        self.Q_dict = {}  # initialize the Q_dictionary that collects the Q_values for each visited (State, Action)
        # Initialize  variables for the parameters
        self.alpha = 0 # Initialize the learning rate
        self.gamma = 0 # Initialize gamma
        self.epsilon = 0.5 # Initialize epsilon
        self.parameters =[] # Initialize the set of parameters epsilon, alpha, gamma
        # Initialize variables for perfomance measurements
        self.sum_rewards = []  # Initialize the rewards count for each of the la 10 trials
        self.penalties = 0 # Initialize the penalties count for each of the la 10 trials
        self.success = 0 # Initialize the count of successful attempts in the last 10 trials
        self.trial = 0  # Initialize the count of trials

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset variables here as required
        self.state = () # Reset the LearningAgent state value when a new trip is initiated
        self.epsilon = 0.3 # Reset epsilon
        self.sum_rewards = [] # Reset the rewards count for each trial
        self.penalties = 0 # Reset the penalties count for each of the la 10 trials
        self.trial +=1 # Counter for the trials
        self.var_parameter()
     

    def var_parameter(self):
        # Function setting the parameters epsilon, alpha and gamma
        # Define when random exploration stops
        if self.parameters[0] == 1:
            self.epsilon_stop = 60
        else:
            self.epsilon_stop = 80

        # Value of epsilon for each trial
        if self.trial < self.epsilon_stop:
            self.epsilon = self.epsilon - self.epsilon * (self.trial-1.) / self.epsilon_stop 
        else:
            self.epsilon = 0.

        # Value of alpha for each trial
        if self.parameters[1] == 3:
            self.alpha = 0.6
        else:
            self.alpha = 0.2

        # Value of alpha for each trial
        if self.parameters[2] ==5:
            self.gamma = 0.6 - 0.6 * (self.trial-1.) / 100.
        else:
            self.gamma = 0.2 - 0.2 * (self.trial-1.) / 100.

        return


    def update(self, t):

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        # To respect the traffic rules, the LearningAgent is required to know the following parameters of the traffic state: light, upcoming, left.
        # To take into account the planner and therefore be able to reach consistently the target, the next_waypoint is also required.
        # The state is recorded in a tuple since it is hashable in the Q_values dictionary.
        self.state = inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint

        # Select action according to your policy 
        # Implement a basic driving agent
        #action = random.choice(self.valid_actions) # Random action for the first step of the project

        # Apply the Exploration/Exploitation policy using epsilon
        if random.random() >= self.epsilon: # EXPLORATION vs EXPLOITATION with epsilon factor
            i = 0
            Q_value = -100
            good_action = None
            good_actions = []
            for action in self.env.valid_actions: 
                if (self.state, action) in self.Q_dict:
                    if self.Q_dict[(self.state, action)] > Q_value: # Test if the action has the best Q_value at this step
                        Q_value = self.Q_dict[(self.state, action)]
                        good_action = action
                        good_actions = [action]
                        i = 0
                    elif self.Q_dict[(self.state, action)] == Q_value: # Test if the action has a Q_value equal to the best at this step
                        i += 1
                        good_actions.append(action)
                else:
                    i += 1
                    if Q_value < 0: # Unexplored actions have a standard Q_value of 0
                        Q_value = 0
                        good_action = action
                        good_actions = [action]
                    elif Q_value == 0:
                        good_actions.append(action)
            

            if i > 0: # If the best options are unexplored actions, choose next action randomly between these options
                if Q_value == 0:
                    good_action = random.choice(good_actions)

        else:
            good_action = random.choice(self.env.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, good_action)
        
        # Performance assessment
        if self.trial > 90:
            if reward < 3:
                self.sum_rewards.append(reward)
                if reward < -0.5:
                    self.penalties += 1
            if reward > 3:
                self.sum_rewards.append(reward-10.)
                self.success += 1                

        # Learn policy based on state, action, and reward
        self.previous_state = self.state # The previous state is recorded
        self.next_waypoint = self.planner.next_waypoint() # From route planner
        inputs = self.env.sense(self) # Update the state after the action
        self.state = inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint # Selection of the required inputs for the next state

        # Select the best action according to your policy for the next step
        Q_value = -100
        next_best_action = None
        for action in self.env.valid_actions:
            if (self.state, action) in self.Q_dict:
                if self.Q_dict[(self.state, action)] > Q_value:
                    Q_value = self.Q_dict[(self.state, action)]
                    next_best_action = action
            else:
                if Q_value < 0:
                    Q_value = 0
                    next_best_action = action

        # Calculate Q_value
        # Formula Q(state, action) = (1-alpha) * Q(state, action) + alpha*(R(s) + gamma* max(Q(next_state, next_action))
        if (self.state, next_best_action) in self.Q_dict:
            if (self.previous_state, good_action) in self.Q_dict:
                self.Q_dict[(self.previous_state, good_action)] = (1 - self.alpha) * self.Q_dict[(self.previous_state, good_action)] + self.alpha * (reward + self.gamma * self.Q_dict[(self.state, next_best_action)])
            else:
                self.Q_dict[(self.previous_state, good_action)] = self.alpha * (reward + self.gamma * self.Q_dict[(self.state, next_best_action)])
        else:
            if (self.previous_state, good_action) in self.Q_dict:
                self.Q_dict[(self.previous_state, good_action)] = (1 - self.alpha) * self.Q_dict[(self.previous_state, good_action)] + self.alpha * (reward + self.gamma * 0)
            else:
                self.Q_dict[(self.previous_state, good_action)] = self.alpha * reward

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    # Lists to record the agent performance (success rates, average reward, rate of penalties)
    success_rates = []
    average_rewards = []
    penalties = []
    numb_states_visited = []
    Dict = {}

    # Repeat the run over the set of parameters (8 combinations here, should be adapted)
    for epsilon in [1,2]: # [1,2]
        for alpha in [3,4]: # [3,4]
            for gamma in [5,6]: # [5,6]

                for i in range(100): 
                    # Set up environment and agent
                    e = Environment()  # create environment (also adds some dummy traffic)
                    a = e.create_agent(LearningAgent)  # create agent
                    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track, enforce deadline or not
                    a.parameters = [epsilon, alpha, gamma]
                    sim = Simulator(e, update_delay=0.0001, display=False) # create simulator (uses pygame when display=True, if available)
                    sim.run(n_trials=100)  # run for a specified number of trials 
                    
                    # Record the performance for the last 10 trials of each run
                    success_rates.append(float(a.success)/10.)
                    average_rewards.append(float(sum(a.sum_rewards))/float(len(a.sum_rewards)))
                    numb_states_visited.append(len(a.Q_dict.keys()))
                    penalties.append(float(a.penalties)/float(len(a.sum_rewards)))
                
                # print a dict of the performances for each combination of parameters
                Dict[(epsilon, gamma, alpha)]= (sum(success_rates)/100., sum(average_rewards)/100., sum(numb_states_visited)/100., float(sum(penalties)/100.))
                success_rates = []
                average_rewards = []
                numb_states_visited = []
                penalties = []

    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "Dict:"
    pprint.pprint(Dict, width=2)

if __name__ == '__main__':
    run()
