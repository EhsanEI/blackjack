import agent
import numpy as np

class MCAgent(agent.BaseAgent):
    """Implements the agent for an RL-Glue environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    def agent_init(self, agent_info= {}):
        """Setup for the agent called when the experiment first starts."""
        self.num_states = agent_info['num_states']
        self.num_actions = agent_info['num_actions']
        self.discount = agent_info['discount']
        self.q = np.zeros((self.num_states,self.num_actions))
        self.sum = np.zeros((self.num_states,self.num_actions))
        self.count = np.zeros((self.num_states,self.num_actions))
        self.random = np.random.RandomState(agent_info['seed'])

    
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sa = [] # List of S,A for the episode
        self.r = [] # List of R for the episode

        action = self.random.randint(self.num_actions) # Exploring Starts

        self.sa.append((observation,action))

        # print('start', self.obs_to_state(observation), action)

        return action


    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        action = self.argmax(self.q[observation,:])

        self.r.append(reward)
        self.sa.append((observation,action))

        # print('step', self.obs_to_state(observation), action)

        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
        self.r.append(reward)
        # print('reward: ', reward)
        # print('----')

        g = 0.0

        sar = list(zip(self.sa, self.r))

        for sa, r in reversed(sar):
            s, a = sa
            # print(self.obs_to_state(s), a, r, g)
            g = self.discount*g + r
            count = self.count[s,a]
            self.count[s,a] += 1
            self.sum[s,a] += g
            self.q[s,a] = self.sum[s,a]/self.count[s,a]

    
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        if message == 'q':
            return self.q
        if message == 'pi':
            return self.q.argmax(axis=1)
        if message == 'count':
            return self.count


    def argmax(self, values):
        return self.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

    # Turns linearized observation back to dict. Only for debugging.
    def obs_to_state(self, obs):
        state = {}
        state['usable_ace'] = int(np.floor(obs/100))
        state['player_sum'] = int(np.floor((obs%100)/10)) + 12
        state['dealer_card'] = (obs%100)%10 + 1
        return state
