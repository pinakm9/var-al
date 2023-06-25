
import os, sys
from pathlib import Path
script_dir = Path(os.path.dirname(os.path.abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
print(module_dir)

import tensorflow as tf 
import arch 
import integrator as it
import time
import pandas as pd
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


DTYPE = 'float32'


def helix_boundary(u, t):
    return u(tf.ones_like(t), t) - t


tb = [-2.*np.pi, 2.*np.pi]
root2 = np.sqrt(2, dtype=DTYPE)
objective = ((tb[1] - tb[0]) / 2.) * (root2 + np.log(1.+root2))


def domain_sampler(n_sample, low=[0., tb[0]], high=[1., tb[1]]):
    r = tf.random.uniform(shape=(n_sample, 1), minval=low[0], maxval=high[0], dtype=DTYPE)
    t = tf.random.uniform(shape=(n_sample, 1), minval=low[1], maxval=high[1], dtype=DTYPE)
    return r, t


gl = it.Gauss_Legendre_2D(domain = [[0., tb[0]], [1., tb[1]]], num=20, dtype=DTYPE, d=4)
r0, t0, w0 = tf.convert_to_tensor(gl.x, dtype=DTYPE), tf.convert_to_tensor(gl.y, dtype=DTYPE), tf.convert_to_tensor(gl.w, dtype=DTYPE)


@tf.function
def area(u):
    with tf.GradientTape() as tape:
        tape.watch([r0, t0])
        u_ = u(r0, t0)
    u_r, u_t = tape.gradient(u_, [r0, t0])
    return tf.reduce_sum(tf.sqrt((1.0 + u_r*u_r)*r0*r0 + u_t*u_t) * w0)


def true(r, t):
    return 0.*r + t

# print(area(true)-objective)
# exit()

@tf.function
def L2_error(u):
    f = (u(r0, t0) - true(r0, t0))**2 * r0 
    return tf.sqrt(tf.reduce_sum(f * w0) / tf.reduce_sum(r0 * w0))


gl = it.Gauss_Legendre(domain = [tb[0], tb[1]], num=10, dtype=DTYPE, d=2)
tb, wb = tf.convert_to_tensor(gl.nodes.reshape(-1, 1), dtype=DTYPE), tf.convert_to_tensor(gl.weights.reshape(-1, 1), dtype=DTYPE)

@tf.function
def constraint_error(u):
    f = (helix_boundary(u, tb))**2
    return tf.sqrt(tf.reduce_sum(f * wb) / tf.reduce_sum(wb))

"""
Things to track: 1) area or main loss, 2) boundary loss, 3) runtime, 4) iteration, 5) save every 100 steps
                 6) total loss
"""

class Solver:

    def __init__(self, save_folder, model_path=None) -> None:
        self.net =  arch.VanillaNet(50, 3, DTYPE, name='helicoid')
        self.save_folder = save_folder
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()
    

    def error(self):
        e = L2_error(self.net)
        o = area(self.net)
        ce = constraint_error(self.net)
        return e.numpy(), o.numpy(), (o/objective-1.).numpy(), ce.numpy()

    @tf.function
    def train_step(self, r, t, b, g):
        with tf.GradientTape() as tape:
            loss_a = area(self.net) * g
            loss_b = tf.reduce_mean(helix_boundary(self.net, t)**2)
            L = loss_a + 0.5*b*loss_b
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss_a, loss_b, L

    def learn(self, epochs=10000, n_sample=1000):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=epochs, decay_rate=1e-3, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        print("{:>6}{:>12}{:>12}{:>12}{:>18}".format('iteration', 'loss_a', 'loss_b', 'loss', 'runtime(s)'))
        start = time.time()
        log = {'iteration': [], 'beta': [], 'loss_a': [], 'loss_b': [], 'loss': [],  'L2-error': [],\
              'objective': [], 'objective-error': [], 'constraint-error': [], 'runtime': []}
        b = tf.constant(1., dtype=DTYPE)
        g = b/b
        epoch, tau = 0, 10
        r, t = domain_sampler(n_sample)
        while epoch < epochs+1:
            loss_a, loss_b, L = self.train_step(r, t, b, g)
            if epoch % 10 == 0:
                step_details = [epoch, loss_a.numpy(), loss_b.numpy(), L.numpy(), time.time()-start]
                print('{:6d}{:15.6f}{:15.6f}{:12.6f}{:12.4f}'.format(*step_details))
                if epoch % 100 == 0:
                    error, objective, objective_error, ce = self.error()
                    log['iteration'].append(step_details[0])
                    log['loss_a'].append(step_details[1])
                    log['loss_b'].append(step_details[2])
                    log['loss'].append(step_details[3])
                    log['L2-error'].append(error)
                    log['objective'].append(objective)
                    log['objective-error'].append(objective_error)
                    log['constraint-error'].append(ce)
                    log['beta'].append(b.numpy())
                    log['runtime'].append(step_details[4])
                    self.net.save_weights('{}/{}_{}'.format(self.save_folder, self.net.name, epoch))
                
            epoch += 1

            if epoch % 2*tau == 0:
                if b < 10000:
                    b += 10.
                r, t = domain_sampler(n_sample)

            if epoch % 1000==0:
                a_, b_ = loss_a.numpy(), 0.5*b.numpy()*loss_b.numpy()
                c_ = np.ceil(np.log10(b_/a_))
                if a_> 1. and b_ < a_ and c_ < 0.:
                    g = np.float32(10**c_) * tf.ones_like(b)
        pd.DataFrame(log).to_csv('{}/train_log.csv'.format(self.save_folder), index=None)
        self.net.save_weights('{}/{}'.format(self.save_folder, self.net.name))

            
save_folder = '../data/helicoid'
Solver(save_folder=save_folder).learn(epochs=50000, n_sample=int(1e3))