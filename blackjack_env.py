#!/usr/bin/env python

from environment import BaseEnvironment

import numpy as np
from copy import deepcopy

class Environment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    # def __init__(self):


    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        self.dealer_sticks = env_info['dealer_sticks']
        self.random = np.random.RandomState(env_info['seed'])
        self.current_state = None

        self.card_probs = np.ones(10)
        self.card_probs[9] = 4 # Face cards count as 10
        self.card_probs /= self.card_probs.sum()

    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = {}
        self.current_state['usable_ace'] = self.random.randint(2)
        self.current_state['player_sum'] = self.random.randint(12,22)
        self.current_state['dealer_card'] = self.random.randint(1,11)

        self.player_ace_count = self.current_state['usable_ace']

        self.reward_obs_term = (0.0, self.observation(self.current_state), False)

        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns: 
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        if action == 0: # Hit

            new_state = deepcopy(self.current_state)
            reward = 0
            terminal = False
            

            # new_card = self.random.choice(range(1,11), p=self.card_probs)
            new_card = min(self.random.randint(1,14), 10)
            # print('new card:', new_card)
            
            if new_card == 1:
                self.player_ace_count += 1
                new_state['player_sum'] = self.current_state['player_sum'] + 11   
            else:
                new_state['player_sum'] = self.current_state['player_sum'] + new_card

            while new_state['player_sum'] > 21 and self.player_ace_count > 0:
                self.player_ace_count -= 1
                new_state['player_sum'] -= 10

            new_state['usable_ace'] = int(self.player_ace_count > 0)

            if new_state['player_sum'] > 21: # Goes bust
                reward = -1
                terminal = True

        elif action == 1: # Stick

            new_state = deepcopy(self.current_state)
            terminal = True

            if self.current_state['dealer_card'] == 1:
                dealer_ace = 1
                dealer_sum = 11
            else:
                dealer_ace = 0
                dealer_sum = self.current_state['dealer_card']

            first_two_cards = True
            while dealer_sum < self.dealer_sticks or first_two_cards:
                first_two_cards = False
                # new_card = self.random.choice(range(1,11), p=self.card_probs)
                new_card = min(self.random.randint(1,14), 10)
                if new_card == 1:
                    dealer_sum += 11
                    dealer_ace += 1
                else:
                    dealer_sum += new_card

                while dealer_sum > 21 and dealer_ace > 0:
                    dealer_sum -= 10
                    dealer_ace -= 1
                dealer_ace = int(dealer_ace > 0)
                # print('dealer:', new_card)

            # print('dealer sum:', dealer_sum)
            if dealer_sum > 21:
                reward = 1
            else:
                if new_state['player_sum'] > dealer_sum:
                    reward = 1
                elif new_state['player_sum'] < dealer_sum:
                    reward = -1
                else:
                    reward = 0
                # reward = int(new_state['player_sum'] > dealer_sum) - int(new_state['player_sum'] < dealer_sum)

        else:
            raise Exception("Invalid action.")

        self.current_state = new_state

        self.reward_obs_term = (reward, self.observation(self.current_state), terminal)

        return self.reward_obs_term

    def observation(self, state):
        # Cases:
        # usable_ace: 2
        # player_sum: 10
        # dealer_card: 11
        return state['usable_ace']*(100) + (state['player_sum']-12)*10 + (state['dealer_card']-1)

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_obs_term[0])

        # else
        return "I don't know how to respond to your message"
