import numpy as np
import matplotlib.pyplot as plt
from rl_glue import RLGlue
from MC_agent import MCAgent
from blackjack_env import Environment
from tqdm import tqdm

inp = 17

seed = 0
agent_info = {'num_actions': 2, 'num_states': 2*10*10, 'discount': 1.0, 'seed': seed}
env_info = {'dealer_sticks': inp, 'seed': seed}


num_episodes = 10#1000000

rl_glue = RLGlue(Environment, MCAgent)
rl_glue.rl_init(agent_info, env_info)
for episode in tqdm(range(num_episodes)):
    rl_glue.rl_episode(0)

pi = rl_glue.agent.agent_message('pi')
q = rl_glue.agent.agent_message('q')
count = rl_glue.agent.agent_message('count')

# print()
# state = {}
# for usable_ace in [0,1]:
#     state['usable_ace'] = usable_ace 
#     for dealer_card in range(1,11):
#         state['dealer_card'] = dealer_card
#         for player_sum in range(12,22):
#             state['player_sum'] = player_sum
#             print(state, pi[rl_glue.environment.observation(state)])
#         print('--')

