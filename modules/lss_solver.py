import tensorflow as tf 
import arch  
import time
import pandas as pd
import numpy as np
import csv

class LogSteadyStateSolver:

    def __init__(self, num_nodes, num_blocks, dtype, name, diff_log_op, optimizer, domain, model_path=None) -> None:
        self.net =  arch.LSTMForgetNet(num_nodes, num_blocks, dtype, name)
        self.diff_log_op = diff_log_op 
        self.domain = domain
        self.dim = len(domain[0])
        self.dtype = dtype
        self.optimizer = optimizer
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()

    def sampler(self, n_sample, domain=None):
        if domain is None:
            domain = self.domain
        X = tf.random.uniform(shape=(n_sample, self.dim), minval=domain[0], maxval=domain[1], dtype=self.dtype)
        return tf.split(X, self.dim, axis=1)

    def loss(self, *args):
        return tf.reduce_mean(self.diff_log_op(self.net, *args)**2)

    @tf.function
    def train_step(self, *args):
        with tf.GradientTape() as tape:
            L = self.loss(*args)
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return L

    def learn(self, epochs=10000, n_sample=1000, save_folder='data', save_along=None, stop_saving=10000):
        args = self.sampler(n_sample)
        print("{:>6}{:>12}{:>18}".format('Epoch', 'Loss', 'Runtime(s)'))
        start = time.time()
        with open('{}/train_log.csv'.format(save_folder), 'w') as logger:
            writer = csv.writer(logger)
            for epoch in range(epochs):
                L = self.train_step(*args)
                if epoch % 10 == 0:
                    step_details = [epoch, L.numpy(), time.time()-start]
                    print('{:6d}{:12.6f}{:18.4f}'.format(*step_details))
                    writer.writerow(step_details)
                    args = self.sampler(n_sample)
                    self.net.save_weights('{}/{}'.format(save_folder, self.net.name))
                if save_along is not None:
                    if epoch <= stop_saving:
                        if epoch % save_along == 0:
                            self.net.save_weights('{}/{}'.format(save_folder, self.net.name + '_' + str(epoch)))
            
