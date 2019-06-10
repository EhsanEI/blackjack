import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import numpy as np
from matplotlib import colors



def observation(state):
    # Cases:
    # usable_ace: 2
    # player_sum: 10
    # dealer_card: 11
    return state['usable_ace']*(100) + (state['player_sum']-12)*10 + (state['dealer_card']-1)


# inp = 17


xticks = np.array(range(10))
yticks = np.array(range(10))

# matplotlib.rcParams.update({'font.size': 14})
matplotlib.rc('axes', labelsize=14, titlesize=18) 
# matplotlib.rc('ytick', labelsize=14)
# matplotlib.rc('title', labelsize=17)  

for inp in range(12,21):
    with open('results/q_'+str(inp)+'.pkl','rb') as f:
        q = pickle.load(f)

    plt.clf()
    fig = plt.figure(figsize=(7,7))
    cmap = colors.ListedColormap(['skyblue', 'tomato'])

    state = {}
    for usable_ace, title, position in [(1,'Usable ace', 211),(0, 'No usable ace', 212)]:
        state['usable_ace'] = usable_ace 
        plot_data = np.zeros((10,10))
        for dealer_card in range(1,11):
            state['dealer_card'] = dealer_card
            for player_sum in range(12,22):
                state['player_sum'] = player_sum
                obs = observation(state) 
                plot_data[player_sum-12, dealer_card-1] = q[obs].argmax()

        print(inp, usable_ace)
        print(plot_data)
        print('----')
        plt.subplot(position)
        plt.pcolormesh(plot_data, vmin=-0.001, vmax=1.001, cmap=cmap)
        plt.title(title)

        ax = plt.gca()
        
        ax.set_xticks(xticks)
        ax.set_xticklabels('')
        ax.set_xticks(xticks+0.5, minor=True)
        ax.set_xticklabels(['A',]+[str(i+1) for i in xticks[1:]], minor=True)

        ax.set_yticks(yticks)
        ax.set_yticklabels('')
        ax.set_yticks(yticks+0.5, minor=True)
        ax.set_yticklabels([str(i+12) for i in xticks], minor=True)
            

        ax.tick_params('both', length=0, which='minor')

        plt.xlabel('Dealer showing')
        plt.ylabel('Player\nsum', rotation=0, labelpad=30)


    cax = plt.axes([0.75, 0.2, 0.075, .5])
    cb = plt.colorbar(cax=cax)
    cb.set_ticks([0.25, 0.75])
    cb.set_ticklabels(['HIT', 'STICK'])
    cb.ax.tick_params(length=0, labelsize=14) 


    plt.subplots_adjust(bottom=0.0, right=0.7, top=1.0, hspace=.3)
    # plt.show()
    plt.savefig('plots/plt_'+str(inp)+'.png', bbox_inches='tight')