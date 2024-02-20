import torch
import random
import numpy as np 
from snake_game import SnakeGame, Direction, Point
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
learning_rate = 0.001

class Agent:
    self __init__(self):
        self.n_games = 0 
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # 
 
    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, gameover):
        pass

    def train_long_memory(self):
        pass
    def train_short_memory(self, state, action, reward, next_state, gameover):
        pass

    def get_action(self, state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0 
    redord = 0 
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state 
        state_old = agent.get_state(game)
        #get move 
        final_move = agent.get_action(state_old)
        #perform move and get new state 
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        #train short memory 
        agent.train_short_memory(state_old, state_new, reward, state_new, gameover)
        #remember 
        agent.remember(state_old, state_new, reward, state_new, gameover)

        if gameover:
            # train long memory 


if __name__ = '__main__':
    train()
