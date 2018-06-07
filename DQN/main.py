
import matplotlib.pyplot as plt
import gym
import numpy as np

from environment import Environment
from agent import Agent
from randomAgent import RandomAgent

def get_rand_agent_memory(env, actionsCount):
    randAgent = RandomAgent(actionsCount)
    while randAgent.memory.is_full(): 
        env.run(randAgent) 

    return randAgent.memory
    

def init_CartPole():
    CartPoleProb = "CartPole-v0"
    env = Environment(CartPoleProb, normalize=False, render=False)

    stateDataCount = env.env.observation_space.shape[0]
    actionsCount = env.env.action_space.n

    print("\nState Data Count:", stateDataCount)
    print("Action Count:", actionsCount)

    agent = Agent(stateDataCount, actionsCount, min_eps=0.01)
    agent.memory = get_rand_agent_memory(env, actionsCount)

    return agent, env


#def run_alg(stateDataCount, actionsCount):


def init_MountainCar():
    MountainProb ="MountainCarContinuous-v0"
    env = Environment(MountainProb, normalize=True, render=True)

    stateDataCount = env.env.observation_space.shape[0]
    actionsCount = env.env.action_space.shape[0]

    print("\nState Data Count:", stateDataCount)
    print("Action Count:", actionsCount)

    agent = Agent(stateDataCount, actionsCount, min_eps=0.1)
    agent.memory = get_rand_agent_memory(env, actionsCount)

    return agent, env

def main():
    reward_each_ep = []
    agent, env = init_CartPole()

    print("\nStart")

    #plt.ion()
    #plt.ylabel('rewards')
    #plt.xlabel('episodes')
    #plt.show()
    
    try:
        for iter in range(1000):
            reward_each_ep.append(env.run(agent))

            # if iter % 250 == 0:
            #     plt.plot(reward_each_ep)
            #     plt.draw()
            #     plt.pause(0.001)
    finally:
        agent.brain.model.save(env.name + "-basic.h5")
        #plt.savefig(env.name + ".png")

    print("End\n")

if __name__ == "__main__":
    main()