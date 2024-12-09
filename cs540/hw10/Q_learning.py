import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict

# Model: ChatGPT: I asked it to implement Q-learning given the starter code. Then I filled in the blanks of my code that I had started.

EPISODES =  30000
LEARNING_RATE = .1  # alpha
DISCOUNT_FACTOR = .99  # weird gamma thing
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)
    
    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):
            # new action implementation here
            if random.uniform(0, 1) < EPSILON:
                # if the random action is picked, we want to explore
                action = env.action_space.sample()
            else:
                # if the random action is not picked, we want to exploit. Use a with highest Q-value for s
                max_q_value = float('-inf')
                action = 0  # Default action
                for a in range(env.action_space.n):
                    if Q_table[(obs, a)] > max_q_value:
                        max_q_value = Q_table[(obs, a)]
                        action = a
            
            # same code as before, take a step with the action
            next_obs,reward,terminated,truncated,info = env.step(action)
            done = terminated or truncated
            
            # Now do the bellman equation stuff
            max_q_value_next = float('-inf')
            for a in range(env.action_space.n):  # Find max Q-value for next state using a for loop
                if Q_table[(next_obs, a)] > max_q_value_next:
                    max_q_value_next = Q_table[(next_obs, a)]

            td_target = reward + DISCOUNT_FACTOR * max_q_value_next
            td_error = td_target - Q_table[(obs, action)]
            Q_table[(obs, action)] += LEARNING_RATE * td_error

            # Update episode reward and move to the next state
            episode_reward += reward
            obs = next_obs
        
        EPSILON = EPSILON * EPSILON_DECAY
        
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open(f'Q_TABLE_QLearning.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################