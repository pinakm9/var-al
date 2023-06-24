
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
    with tf.GradientTape() as tape:
        tape.watch([r, t])
        u_ = u(r, t)
    u_r, u_t = tape.gradient(u_, [r, t])
    return tf.sqrt((1.0 + u_r*u_r)*r*r + u_t*u_t)

R = 3.
tb = [np.pi, np.pi]

def boundary(u, t):
    return u(R * tf.ones_like(t), t) - R**2 * tf.cos(2. * t)



root2 = np.sqrt(2, dtype=DTYPE)
objective = ((tb[1] - tb[0]) / 2.) * (root2 + np.log(1.+root2))


def domain_sampler(n_sample, low=[0., tb[0]], high=[R, tb[1]]):
    r = tf.random.uniform(shape=(n_sample, 1), minval=low[0], maxval=high[0], dtype=DTYPE)
    t = tf.random.uniform(shape=(n_sample, 1), minval=low[1], maxval=high[1], dtype=DTYPE)
    return r, t

def true(r, t):
    return t

"""
Things to track: 1) area or main loss, 2) boundary loss, 3) runtime, 4) iteration, 5) save every 100 steps
                 6) total loss
"""

class Solver:

    def __init__(self, save_folder, model_path=None) -> None:
        self.net =  arch.LSTMForgetNet(50, 3, DTYPE, name='Enneper')
        self.save_folder = save_folder
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()
        self.r0, self.t0 = domain_sampler(100000)

    def error(self):
        e = tf.abs(self.net(self.r0, self.t0) - true(self.r0, self.t0)) * self.r0 / ((tb[1] - tb[0]) / 2.)
        o = tf.reduce_mean(minimal_op(self.net, self.r0, self.t0)).numpy() * (tb[1]-tb[0])
        return tf.reduce_mean(e).numpy(), o, (o-objective)

    @tf.function
    def train_step(self, r, t, b):
        with tf.GradientTape() as tape:
            loss_a = tf.reduce_mean(minimal_op(self.net, r, t))
            loss_b = tf.reduce_mean(boundary(self.net, t)**2)
            L = loss_a + b*loss_b
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss_a, loss_b, L

    def learn(self, epochs=10000, n_sample=1000):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=epochs, decay_rate=1e-3, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        print("{:>6}{:>12}{:>12}{:>12}{:>18}".format('iteration', 'loss_a', 'loss_b', 'loss', 'runtime(s)'))
        start = time.time()
        log = {'iteration': [], 'beta': [], 'loss_a': [], 'loss_b': [], 'loss': [],  'L1-error': [],\
              'objective': [], 'objective-error': [], 'runtime': []}
        epsilon = np.pi
        low, high = [0., tb[0]-epsilon], [0., tb[1]+epsilon]
        r, t = domain_sampler(n_sample, low, high)
        b = tf.constant(100., dtype=DTYPE)
        for epoch in range(epochs+1):
            loss_a, loss_b, L = self.train_step(r, t, b)
            if epoch % 10 == 0:
                step_details = [epoch, loss_a.numpy(), loss_b.numpy(), L.numpy(), time.time()-start]
                print('{:6d}{:15.6f}{:15.6f}{:12.6f}{:12.4f}'.format(*step_details))
                if epoch % 100 == 0:
                    error, objective, objective_error = self.error()
                    log['iteration'].append(step_details[0])
                    log['loss_a'].append(step_details[1])
                    log['loss_b'].append(step_details[2])
                    log['loss'].append(step_details[3])
                    log['L1-error'].append(error)
                    log['objective'].append(objective)
                    log['objective-error'].append(objective_error)
                    log['beta'].append(b.numpy())
                    log['runtime'].append(step_details[4])
                    self.net.save_weights('{}/{}_{}'.format(self.save_folder, self.net.name, epoch))
                    r, t = domain_sampler(n_sample, low, high)
                    b += 0.
        pd.DataFrame(log).to_csv('{}/train_log.csv'.format(self.save_folder), index=None)
        self.net.save_weights('{}/{}'.format(self.save_folder, self.net.name))

            
save_folder = '../data/Enneper'
Solver(save_folder=save_folder).learn(epochs=10000)