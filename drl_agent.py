# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 20:42:50 2021

@author: vincent_liao
"""
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam, RMSprop
from UPM_env_v2 import Factory
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
import time




class DQL_Agent:
    def __init__(self, n_jobs, n_machines, n_actions, n_types):
        # problem data scale
        self.n_jobs     = n_jobs
        self.n_machines = n_machines
        self.n_actions  = n_actions
        self.n_types    = n_types
        
        #hyper-parameters
        self.finish        = False
        self.epsilon       = 1.0 #Initial exploration rate
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995
        self.gamma         = 0.05
        self.batch_size    = 32 #Batch size for replay
        self.max_treward   = 0
        self.averages      = list()
        self.memory        = deque(maxlen=2000)#deque collection for limited history
        
        self.num_update_target_network = 1000
        self.update_per_actions        = 4
#        self.osn           = env.observation_space.shape[0]
        self.hu            = 24
        self.opt           = keras.optimizers.Adam(learning_rate = 1e-3, clipnorm=1.0)
        self.lr            = 0.0001
        self.activation    = "relu"
        
        self.frame_count   = 0
        self.loss_function = keras.losses.Huber()
        self.loss = None
        
        self.q_network, self.target_q_network = self._build_model()
        self.q_network.summary()
        
    def _build_model(self):
        # Network architecture
        input_m1 = keras.Input(shape=(self.n_jobs, 6, 1), dtype=tf.float32)
        input_m2 = keras.Input(shape=(self.n_machines, 4, 1), dtype=tf.float32)
        input_m3 = keras.Input(shape=(self.n_types, self.n_types, 1), dtype=tf.float32)

        layer1 = layers.Conv2D(16, 3, strides=1, activation=self.activation)(input_m1)
        layer1 = layers.Conv2D(16, 3, strides=1, activation=self.activation)(layer1)
        layer1 = layers.Flatten()(layer1)

        layer2 = layers.Conv2D(16, 2, strides=1, activation=self.activation)(input_m2)
        layer2 = layers.Conv2D(16, 2, strides=1, activation=self.activation)(layer2)
        layer2 = layers.Flatten()(layer2)

        layer3 = layers.Conv2D(16, 3, strides=1, activation=self.activation)(input_m3)
        layer3 = layers.Conv2D(16, 3, strides=1, activation=self.activation)(layer3)
        layer3 = layers.Flatten()(layer3)

        combined = layers.concatenate([layer1, layer2, layer3])
        common = layers.Dense(units = 64, activation = self.activation)(combined)
        common = layers.Dense(units = 64, activation = self.activation)(common)
        q_value = layers.Dense(units = self.n_actions)(common)
        
        q_network        = keras.Model(inputs = [input_m1, input_m2, input_m3], outputs = q_value )
        target_q_network = keras.Model(inputs = [input_m1, input_m2, input_m3], outputs = q_value )
        
        return q_network, target_q_network
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return np.random.choice(self.n_actions)
        
        action = self.q_network([state[0], state[1], state[2]])
        return np.argmax(action)
    
    def train_q_network(self):
#        self.memory.append([state, action, reward,
#                                    next_state, done])
        # train per 4 actions
        indices = np.random.choice(range(len(self.memory[4])), size = self.batch_size)

        # sample
        s_m1_sample = np.array([self.memory[i][0][0][0] for i in indices])
        s_m2_sample = np.array([self.memory[i][0][1][0] for i in indices])
        s_m3_sample = np.array([self.memory[i][0][2][0] for i in indices])
        
        s_next_m1_sample = np.array([self.memory[i][3][0][0] for i in indices])
        s_next_m2_sample = np.array([self.memory[i][3][1][0] for i in indices])
        s_next_m3_sample = np.array([self.memory[i][3][2][0] for i in indices])
        
        rewards_sample = [self.memory[i][2] for i in indices]
        action_sample  = [self.memory[i][1] for i in indices]
        done_sample    = tf.convert_to_tensor([float(self.memory[i][4]) for i in indices])

        # q(next_state)
        future_rewards = self.target_q_network.predict(
            [s_next_m1_sample, s_next_m2_sample, s_next_m3_sample]
        )
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                    future_rewards, axis=1
        )
        # set last q value to -1
        updated_q_values = updated_q_values*(1 - done_sample) - done_sample
        masks = tf.one_hot(action_sample, self.n_actions)

        with tf.GradientTape() as tape:
          # Train the model on the states and updated Q-values
          q_values = self.q_network([s_m1_sample, s_m2_sample, s_m3_sample])
          # only update q-value which is chosen
          q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
          # calculate loss between new Q-value and old Q-value
          self.loss = self.loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(self.loss, self.q_network.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))
    
    def update_target_network(self):
        # update per update_target_network steps
        self.target_q_network.set_weights(self.q_network.get_weights())
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        else:
            self.epsilon == self.epsilon_min
    
    def learn(self, episodes, n_steps, writer):
        episode_reward = []
        total_step = 0
        
        for e in range(1, episodes + 1):
            obs = env.reset()
            treward = 0
        
            for _ in range(n_steps + 1):
                self.frame_count += 1
                total_step += 1
                
                # reshape to CNN acceptable
                obs_m1 = obs[0].reshape(1, self.n_jobs, 6, 1)
                obs_m2 = obs[1].reshape(1, self.n_machines, 4, 1)
                obs_m3 = obs[2].reshape(1, self.n_types, self.n_types, 1)
                
                state = [obs_m1, obs_m2, obs_m3]
                
                action = self.act(state)
                
                
                next_obs, reward, done, info = env.step(action)
                
                treward += reward
                
                next_obs_m1 = next_obs[0].reshape(1, self.n_jobs, 6, 1)
                next_obs_m2 = next_obs[1].reshape(1, self.n_machines, 4, 1)
                next_obs_m3 = next_obs[2].reshape(1, self.n_types, self.n_types, 1)

                next_state = [next_obs_m1, next_obs_m2, next_obs_m3]
               
                self.memory.append([state, action, reward,
                                    next_state, done])
                obs = next_obs
                
                if self.frame_count % self.update_per_actions == 0 and len(self.memory) >= agent.batch_size:
                    self.train_q_network()
                
                if self.frame_count % self.num_update_target_network == 0:
                    self.update_target_network()
            
                if done:
                    writer.add_scalar('Train-Episode/Reward', treward,
                                      e)
                    writer.add_scalar('Train-Episode/Makespan', env.makespan,
                                      e)
                    writer.add_scalar('Train-Episode/Epsilon', self.epsilon,
                                      e)

#                    if self.loss is not None:
#                        writer.add_scalar('Train-Step/Loss', self.loss,
#                                          total_step)
                if done:
                    break
               
            self.decay_epsilon()
            
            episode_reward.append(treward)
            #average of last 25 episodes total_rewards
            av = sum(episode_reward[-25:]) / 25
            self.averages.append(av)
            self.max_treward = max(self.max_treward, treward)
            templ = 'episode: {:4d}/{} | treward: {:4.2f} | '
            templ += 'av: {:4.2f} | max: {:4.2f} |'
            print(templ.format(e, episodes, treward, 
                               av, self.max_treward))
            
            #save model every 500 episodes
            if e % 500 == 0:
                dir1 = 'model/q_network_' + str(e)
                dir2 = 'model/t_q_network_' + str(e)
                self.q_network.save(dir1)
                self.target_q_network.save(dir2)           

        plt.plot(episode_reward)
        env.gantt_plot.draw_gantt(350)
#            if av < optimal_makespan and self.finish:
#                print()
#                break

    def test(self, episodes, n_steps):
        episode_reward = []
        
        for e in range(1, episodes + 1):
            obs = env.reset()
            treward = 0
            
            while True:
                obs_m1 = obs[0].reshape(1, n_jobs, 6, 1)
                obs_m2 = obs[1].reshape(1, n_machines, 4, 1)
                obs_m3 = obs[2].reshape(1, n_types, n_types, 1)
        
                action = np.argmax(self.q_network.predict([obs_m1, obs_m2, obs_m3])[0])
                print("action: ", action)
                next_obs, reward, done, info = env.step(action)
                episode_reward.append(reward)
                obs = next_obs
                
                treward += reward
                
                if done:
                    print('Makespan: ',info)
                    break
            
            episode_reward.append(treward)
        
        print("average reward = ",np.average(episode_reward))
    
if __name__ == "__main__":
    n_jobs     = 15
    n_machines = 3
    n_actions  = 4
    n_types    = n_jobs

    n_episodes            = 3000 
    max_steps_per_episode = 2000
    
    # Tensorboard to trace the learning process
    writer = SummaryWriter(f'log/DQN-{time.time()}')
    
    #environment
    env = Factory()
    
    agent = DQL_Agent(n_jobs, n_machines, n_actions, n_types)
    agent.learn(n_episodes, max_steps_per_episode, writer)
    agent.test(30, max_steps_per_episode)
    
    writer.close()
    
    
    
    