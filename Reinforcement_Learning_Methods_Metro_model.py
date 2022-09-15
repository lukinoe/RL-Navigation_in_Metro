import numpy as np

#Hyperparameters
SMALL_ENOUGH = 0.005
GAMMA = 0.9
TRANSITION_PROB = 0.80  # transitition probability of taking the best action max_a    


#Define all states
all_states=[]   
for i in range(21):
    all_states.append(i)


rewards = np.zeros(21)
rewards[:] = -0.3 # penalty for each action
rewards[3] = -5   # Shitty place, mood is down
rewards[10] = 3   # meet a friend
rewards[15]=  5   # Goal


#Dictionnary of possible actions. L = Walk_Left, R = Walk_Right, M = Metro, C = Car
actions = {                   
    (0): ('R', 'M1'), 
    (1):('L', 'R'),    
    (2):('L', 'R'), 
    (3):('L', 'R'), 
    (4):('L', 'R'), 
    (5):('L', 'R'), 
    (6):('L', 'R'), 
    (7):('L', 'R', 'M1'), 
    (8):('L', 'R'), 
    (9):('L', 'R'), 
    (10):('L', 'R'), 
    (11):('L', 'R'), 
    (12):('L', 'R'), 
    (13):('L', 'R'), 
    (14):('L', 'R'), 
    (15):('L', 'R', 'S'), 
    (16):('L', 'R'), 
    (17):('L', 'R', 'M2'), 
    (18):('L', 'R'), 
    (19):('L', 'R'), 
    (20):('L', 'M2'), 
}

#Define an initial policy
policy={}
for s in actions.keys():
    policy[s] = np.random.choice(actions[s])

#Define initial value function 
V={}
for s in all_states:
    if s in actions.keys():
        V[s] = 0


def step(s,a):
    if a == 'M1':
        nxt = s + 5
    if a == 'M2':
        nxt = s - 6
    if a == 'L':
        nxt = s -1
    if a == 'R':
        nxt = s + 1
    if a == 'S':
        nxt = s

    return nxt


'''Value Iteration'''

iteration = 0

while True:
    biggest_change = 0
    for s in all_states:            
        if s in policy:
            
            old_v = V[s]
            new_v = 0
            
            for a in actions[s]:

                #Choose a new random action to do 
                a_rand = np.random.choice([i for i in actions[s] if i != a])

                nxt = step(s,a)                
                nxt_rand = step(s,a_rand)

                V_s_1_ALL_STATES_sum = sum([V[step(s,a_)] for a_ in actions[s] if i != a])

                TRANSITION_PROB_ = TRANSITION_PROB
                TRANSITION_PROB_WALK = 1
                TRANSITION_PROB_METRO = 0.66

                TRANSITION_PROB_ = TRANSITION_PROB_WALK if a == "R" or a == "L" or "S" else TRANSITION_PROB_METRO


                '''ATTENTION: TYPICALLY YOU WOULD SUM THE VALUES OVER ALL NEXT STATES, NOT JUST ONE RANDOM STATE'''
                v = rewards[s] + (GAMMA * (TRANSITION_PROB_ * V[nxt] + ((1 - TRANSITION_PROB_) * V[nxt_rand]))) 

                if v > new_v: #Is this the best action so far? If so, keep it
                    new_v = v
                    policy[s] = a

       #Save the best of all actions for the state                                
            V[s] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))

            
   #See if the loop should stop now         
    if biggest_change < SMALL_ENOUGH:
        break
    print(iteration)
    iteration += 1

print(V, policy)



