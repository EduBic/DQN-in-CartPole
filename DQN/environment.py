
import gym
import numpy as np

class Environment:

    def __init__(self, problem, normalize=False, render=False):
        self.normalize_on = normalize
        self.render_on = render
        self.name = problem

        print("Normalize:", self.normalize_on, 
                "Render:", self.render_on)

        self.env = gym.make(problem)

        high = self.env.observation_space.high
        low = self.env.observation_space.low

        self.mean = (high + low) / 2
        self.spread = abs(high - low) / 2

    def normalize(self, state):
        return (state - self.mean) / self.spread

    def run(self, agent):
        curr_state = self.env.reset()
        if self.normalize_on:
            curr_state = self.normalize(curr_state)
        
        tot_reward = 0

        done = False
        while not done:
            if self.render_on: 
                self.env.render()

            action = agent.act(curr_state)

            #print("\n action", action)

            # if action == 0:
            #     a_ = 0 # move left
            # elif action == 1:
            #     a_ = 2 # move right

            new_state, reward, done, _ = self.env.step(action)
            if self.normalize_on:
                new_state = self.normalize(new_state)

            if done:
                new_state = None

            agent.observe( (curr_state, action, reward, new_state) )
            # perform a learning step
            agent.replay()

            curr_state = new_state

            tot_reward += reward

        #print("Total reward:", tot_reward)
        return tot_reward

    def set_seed(self, seed):
        self.env.seed(seed)