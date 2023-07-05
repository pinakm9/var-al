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

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


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


gl = it.Gauss_Legendre_2D(domain = [[0., tb[0]], [1., tb[1]]], num=50, dtype=DTYPE, d=2)
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


gl1 = it.Gauss_Legendre(domain = [tb[0], tb[1]], num=100, dtype=DTYPE, d=4)
t1, w1 = tf.convert_to_tensor(gl1.nodes.reshape(-1, 1), dtype=DTYPE), tf.convert_to_tensor(gl1.weights.reshape(-1, 1), dtype=DTYPE)

@tf.function
def constraint_error(u):
    f = (helix_boundary(u, t1))**2
    return tf.sqrt(tf.reduce_sum(f * w1) / tf.reduce_sum(w1))

"""
Things to track: 1) area or main loss, 2) boundary loss, 3) runtime, 4) iteration, 5) save every 100 steps
                 6) total loss
"""

class Solver:

    def __init__(self, save_folder, model_path=None) -> None:
        self.net = arch.VanillaNet(50, 3, DTYPE, name='helicoid-al')
        self.mul = arch.VanillaNet(50, 3, DTYPE, name='helicoid-al-mul')
        self.save_folder = save_folder
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()

    def error(self):
        e = L2_error(self.net)
        o = area(self.net)
        ce = constraint_error(self.net)
        return e.numpy(), o.numpy(), (o/objective-1.).numpy(), ce.numpy()

    @tf.function
    def train_step(self, r, t, b):
        with tf.GradientTape() as tape:
            loss_a = area(self.net)
            loss_b = tf.reduce_mean(helix_boundary(self.net, t)**2)
            loss_m = tf.reduce_mean(helix_boundary(self.net, t) * self.mul(t))
            L = loss_a + 0.5*b*loss_b + loss_m 
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss_a, loss_b, loss_m, L


    @tf.function
    def train_step_mul(self, mu_0, t, b):
        with tf.GradientTape() as tape:
            L = tf.reduce_mean((self.mul(t) - mu_0 - b * helix_boundary(self.net, t))**2)
        grads = tape.gradient(L, self.mul.trainable_weights)
        self.optimizer_mul.apply_gradients(zip(grads, self.mul.trainable_weights))
        return L

    def learn(self, epochs=10000, n_sample=1000):
        print("{:>6}{:>12}{:>12}{:>12}{:>8}{:>12}{:>18}"\
              .format('iteration', 'loss_a', 'loss_b', 'loss_m', 'loss', 'loss_mul', 'runtime(s)'))
        start = time.time()
        log = {'iteration': [], 'beta': [], 'loss_a': [], 'loss_b': [], 'loss_m': [], 'loss': [], 'loss_mul': [],\
                'L2-error': [], 'objective': [], 'objective-error': [], 'constraint-error': [], 'runtime': []}
        epoch, tau, b0, delb, maxb = 0, 10, 100, 1.01, 5000
        initial_rate = 1e-4
        decay_rate = 2e-1
        decay_steps = int(2*tau)
        final_learning_rate = 1e-4
        final_decay_rate = 1e-1
        drop = 1.
        tipping_point = int(2*tau*(maxb-b0)/delb)
        final_decay_steps = epochs - tipping_point
        lr_schedule = arch.CyclicLR(initial_rate, decay_rate, decay_steps,final_learning_rate, final_decay_rate, final_decay_steps,\
                            drop, tipping_point)
        lr_schedule_mul = arch.CyclicLR(initial_rate, decay_rate, decay_steps,final_learning_rate, final_decay_rate, final_decay_steps,\
                            drop, tipping_point)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.optimizer_mul = tf.keras.optimizers.Adam(learning_rate=lr_schedule_mul)
        b = tf.constant(b0, dtype=DTYPE)
        r, t = domain_sampler(n_sample)
        flag = True
        while epoch < epochs+1:
            for _ in range(tau):
                loss_a, loss_b, loss_m, L = self.train_step(r, t, b)
            mu_0 = self.mul(t)
            for _ in range(tau):
                loss_mul = self.train_step_mul(mu_0, t, b)
            if epoch % 10 == 0:
                step_details = [epoch, loss_a.numpy(), loss_b.numpy(), loss_m.numpy(), L.numpy(), loss_mul.numpy(), time.time()-start]
                print('{:6d}{:15.6f}{:15.6f}{:10.4f}{:10.4f}{:15.4f}{:12.4f}'.format(*step_details))
                if epoch % 100 == 0:
                    error, objective, objective_error, ce = self.error()
                    log['iteration'].append(step_details[0])
                    log['loss_a'].append(step_details[1])
                    log['loss_b'].append(step_details[2])
                    log['loss_m'].append(step_details[3])
                    log['loss'].append(step_details[4])
                    log['loss_mul'].append(step_details[5])
                    log['L2-error'].append(error)
                    log['objective'].append(objective)
                    log['objective-error'].append(objective_error)
                    log['constraint-error'].append(ce)
                    log['beta'].append(b.numpy())
                    log['runtime'].append(step_details[6])
                    self.net.save_weights('{}/{}_{}'.format(self.save_folder, self.net.name, epoch))
            
            epoch += 2*tau

            if b < maxb:
                b *= delb
            r, t = domain_sampler(n_sample)

  
                
        pd.DataFrame(log).to_csv('{}/train_log.csv'.format(self.save_folder), index=None)
        self.net.save_weights('{}/{}'.format(self.save_folder, self.net.name))

            
save_folder = '../data/helicoid-al'
Solver(save_folder=save_folder).learn(epochs=20000, n_sample=int(1e3))
