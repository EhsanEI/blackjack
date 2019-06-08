import matplotlib.pyplot as plt
import pickle
import numpy as np

def observation(state):
    # Cases:
    # usable_ace: 2
    # player_sum: 10
    # dealer_card: 11
    return state['usable_ace']*(100) + (state['player_sum']-12)*10 + (state['dealer_card']-1)


inp = 17

with open('results/q_'+str(inp)+'.pkl','rb') as f:
    q = pickle.load(f)

state = {}
for usable_ace, position in [(0,211),(1,212)]:
    state['usable_ace'] = usable_ace 
    plot_data = np.zeros((10,10))
    for dealer_card in range(1,11):
        state['dealer_card'] = dealer_card
        for player_sum in range(12,22):
            state['player_sum'] = player_sum
            obs = observation(state) 
            plot_data[player_sum-12, dealer_card-1] = q[obs].argmax()

    plt.subplot(position)
    plt.pcolormesh(plot_data)
    plt.title(usable_ace)

plt.show()