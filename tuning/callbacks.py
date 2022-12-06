import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class LerningRateCallback(tf.keras.callbacks.Callback):
    def __init__(self, min_learning_rate=1e-10, max_learning_rate=1e-2):
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.losses = []
        self.current = 0
        self.update = 10
        self.steps = 50
        
    def on_train_begin(self, logs={}):
        self.weights = self.model.get_weights()
        self.lr = np.geomspace(self.min_learning_rate, self.max_learning_rate, self.steps) 
        self.model.optimizer.learning_rate = self.lr[self.current]
        
    def on_batch_end(self, batch, logs={}):
        self.update -= 1
        
        if(self.current==(self.steps-1) and self.update==0):
            self.losses.append(logs.get('loss'))
            self.model.stop_training = True  
        
        if(self.update == 0 and self.current!=(self.steps-1)):
            self.losses.append(logs.get('loss'))
            self.update = 10
            self.current += 1
            self.model.set_weights(self.weights)
            self.model.optimizer.learning_rate = self.lr[self.current]

            
    def on_train_end(self, logs={}):
        sum_loss = []
        for i in range(2,49):
            sum_loss.append((self.losses[i] + self.losses[i-1] + self.losses[i-2])/3)
            
        plt.plot(self.lr, self.losses)
        plt.plot(self.lr[2:49], sum_loss)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.show()