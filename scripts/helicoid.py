
import os, sys
from pathlib import Path
script_dir = Path(os.path.dirname(os.path.abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
print(module_dir)

import tensorflow as tf 
import arch  
import time
import pandas as pd
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


DTYPE = 'float32'

@tf.function
def minimal_op(u, r, t):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch([r, t])
      u_ = u(r, t)
      u_r, u_t = tape.gradient(u_, [r, t])
    u_rr = tape.gradient(u_r, r)
    u_tt = tape.gradient(u_t, t)
  
    return tf.sqrt(r**2 * (1.0 + u_r**2) + u_t**2)


def helix_boundary(u, t):
    return u(tf.ones_like(t), t) - t


def domain_sampler(n_sample, low=[0., 0.], high=[1., 2.0 * np.pi]):
    r = tf.random.uniform(shape=(n_sample, 1), minval=low[0], maxval=high[0])
    t = tf.random.uniform(shape=(n_sample, 1), minval=low[1], maxval=high[1])
    return r, t 



"""
Things to track: 1) area or main loss, 2) boundary loss, 3) runtime, 4) iteration, 5) save every 100 steps
                 6) total loss
"""

class Solver:

    def __init__(self, save_folder, model_path=None) -> None:
        self.net =  arch.LSTMForgetNet(50, 2, DTYPE, name='helicoid')
        learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2000, 10000], [5e-3, 1e-3, 5e-4, 1e-4])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.save_folder = save_folder
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()


    @tf.function
    def train_step(self, r, t, beta):
        with tf.GradientTape() as tape:
            loss_a = tf.reduce_mean(minimal_op(self.net, r, t))
            loss_b = tf.reduce_mean(helix_boundary(self.net, t)**2)
            L = loss_a + beta * loss_b
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss_a, loss_b, L

    def learn(self, epochs=10000, n_sample=1000):
        print("{:>6}{:>12}{:>12}{:>12}{:>18}".format('iteration', 'loss_a', 'loss_b', 'loss', 'runtime(s)'))
        start = time.time()
        log = {'iteration': [], 'loss_a': [], 'loss_b': [], 'loss': [], 'beta': [], 'runtime': []}
        r, t = domain_sampler(n_sample)
        beta = 1.
        for epoch in range(epochs+1):
            loss_a, loss_b, L = self.train_step(r, t, beta)
            if epoch % 10 == 0:
                step_details = [epoch, loss_a.numpy(), loss_b.numpy(), L.numpy(), time.time()-start]
                print('{:6d}{:15.6f}{:15.6f}{:12.6f}{:12.4f}'.format(*step_details))
                log['iteration'].append(step_details[0])
                log['loss_a'].append(step_details[1])
                log['loss_b'].append(step_details[2])
                log['loss'].append(step_details[3])
                log['beta'].append(beta)
                log['runtime'].append(step_details[4])
                r, t = domain_sampler(n_sample)
                if epoch % 100 == 0:
                    self.net.save_weights('{}/{}_{}'.format(self.save_folder, self.net.name, epoch))
            beta += 0.01
        pd.DataFrame(log).to_csv('{}/train_log.csv'.format(self.save_folder), index=None)
        self.net.save_weights('{}/{}'.format(self.save_folder, self.net.name))

            
save_folder = '../data/helicoid'
Solver(save_folder=save_folder).learn(epochs=100)