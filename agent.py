import torch
import random
import numpy as np 
from snake_game import SnakeGame, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
learning_rate = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0 
        self.epsilon = 0 # randomness
        self.gamma = 0.8 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=learning_rate, gamma=self.gamma)
 
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
                # danger straight
                (dir_r and game.is_collision(point_r)) or
                 (dir_l and game.is_collision(point_l)) or 
                 (dir_u and game.is_collision(point_u)) or
                 (dir_d and game.is_collision(point_d)),

                 # danger right
                 (dir_r and game.is_collision(point_d)) or
                 (dir_l and game.is_collision(point_u)) or 
                 (dir_u and game.is_collision(point_r)) or
                 (dir_d and game.is_collision(point_l)),
                
                 # danger left
                 (dir_r and game.is_collision(point_u)) or
                 (dir_l and game.is_collision(point_d)) or 
                 (dir_u and game.is_collision(point_l)) or
                 (dir_d and game.is_collision(point_r)),
                
                 # move direction 
                 dir_l, dir_r, dir_u, dir_d,

                 # food location
                 game.food.x < game.head.x, # food left 
                 game.food.x > game.head.x, # food right 
                 game.food.y < game.head.y, # food up
                 game.food.y > game.head.y, # food down
                ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # return a list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, gameovers = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

    def train_short_memory(self, state, action, reward, next_state, gameover):
        self.trainer.train_step(state, action, reward, next_state, gameover)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1 

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0 
    record = 0 
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state 
        state_old = agent.get_state(game)
        #get move 
        final_move = agent.get_action(state_old)
        #perform move and get new state 
        reward, gameover, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        #train short memory 
        agent.train_short_memory(state_old, final_move, reward, state_new, gameover)
        #remember 
        agent.remember(state_old, final_move, reward, state_new, gameover)

        if gameover:
            # train long memory and plot results
            game.reset()
            agent.n_games += 1 
            agent.train_long_memory()

            # check if the current score is higher than the highest score.
            if score > record:
                record = score
                agent.model.save()

            print(f'game: {agent.n_games}, score: {score}, record: {record}')


if __name__ == '__main__':
    train()
