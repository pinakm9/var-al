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


@tf.function
def minimal_op(u, r, t):
    with tf.GradientTape() as tape:
        tape.watch([r, t])
        u_ = u(r, t)
    u_r, u_t = tape.gradient(u_, [r, t])
    return tf.sqrt((1.0 + u_r*u_r)*r*r + u_t*u_t)


def helix_boundary(u, t):
    return u(tf.ones_like(t), t) - t


tb = [-2.*np.pi, 2.*np.pi]
root2 = np.sqrt(2, dtype=DTYPE)
objective = ((tb[1] - tb[0]) / 2.) * (root2 + np.log(1.+root2))


def domain_sampler(n_sample, low=[0., tb[0]], high=[1., tb[1]]):
    r = tf.random.uniform(shape=(n_sample, 1), minval=low[0], maxval=high[0], dtype=DTYPE)
    t = tf.random.uniform(shape=(n_sample, 1), minval=low[1], maxval=high[1], dtype=DTYPE)
    return tf.sqrt(r), t


gl = it.Gauss_Legendre_2D(domain = [[0., tb[0]], [1., tb[1]]], num=13, dtype=DTYPE, d=7)
r0, t0, w0 = tf.convert_to_tensor(gl.x, dtype=DTYPE), tf.convert_to_tensor(gl.y, dtype=DTYPE), tf.convert_to_tensor(gl.w, dtype=DTYPE)

gl1 = it.Gauss_Legendre_2D(domain = [[0., tb[0]], [1., tb[1]]], num=15, dtype=DTYPE, d=10)
r1, t1, w1 = tf.convert_to_tensor(gl1.x, dtype=DTYPE), tf.convert_to_tensor(gl1.y, dtype=DTYPE), tf.convert_to_tensor(gl1.w, dtype=DTYPE)

@tf.function
def area(u):
    with tf.GradientTape() as tape:
        tape.watch([r0, t0])
        u_ = u(r0, t0)
        u_r, u_t = tape.gradient(u_, [r0, t0])
    return tf.reduce_sum(tf.sqrt((1.0 + u_r*u_r)*r0*r0 + u_t*u_t) * w0)

@tf.function
def area1(u):
    with tf.GradientTape() as tape:
        tape.watch([r1, t1])
        u_ = u(r1, t1)
        u_r, u_t = tape.gradient(u_, [r1, t1])
    return tf.reduce_sum(tf.sqrt((1.0 + u_r*u_r)*r1*r1 + u_t*u_t) * w1)


def true(r, t):
    return t


@tf.function
def L1_error(u):
    f = tf.abs(u(r1, t1) - true(r1, t1)) * r1
    return tf.reduce_sum(f * w1) / ((tb[1] - tb[0]) / 2.)

"""
Things to track: 1) area or main loss, 2) boundary loss, 3) runtime, 4) iteration, 5) save every 100 steps
                 6) total loss
"""

class Solver:

    def __init__(self, save_folder, model_path=None) -> None:
        self.net = arch.VanillaNet(64, 3, DTYPE, name='helicoid-al')
        self.mul = arch.VanillaNet(64, 3, DTYPE, name='helicoid-al-mul')
        self.save_folder = save_folder
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()

    def error(self):
        e = L1_error(self.net)
        o = area1(self.net)
        return e.numpy(), o.numpy(), (o-objective).numpy()

    @tf.function
    def train_step(self, r, t, b):
        with tf.GradientTape() as tape:
            loss_a = area(self.net) * 1e-2
            loss_b = tf.reduce_mean(helix_boundary(self.net, t)**2)
            loss_m = tf.reduce_mean(helix_boundary(self.net, t) * self.mul(t))
            L = loss_a + 0.5*b*loss_b + loss_m
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss_a, loss_b, loss_m, L, b


    @tf.function
    def train_step_mul(self, t, b):
        mu_0 = self.mul(t)
        with tf.GradientTape() as tape:
            L = tf.reduce_mean((self.mul(t) - mu_0 - b * helix_boundary(self.net, t))**2)
        grads = tape.gradient(L, self.mul.trainable_weights)
        self.optimizer_mul.apply_gradients(zip(grads, self.mul.trainable_weights))
        return L

    def learn(self, epochs=10000, n_sample=1000):
        ir, dr = 1e-3, 1e-3
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(ir, decay_steps=epochs, decay_rate=dr, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        lr_schedule_mul = tf.keras.optimizers.schedules.ExponentialDecay(ir, decay_steps=epochs, decay_rate=dr, staircase=False)
        self.optimizer_mul = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        print("{:>6}{:>12}{:>12}{:>12}{:>8}{:>12}{:>18}"\
              .format('iteration', 'loss_a', 'loss_b', 'loss_m', 'loss', 'loss_mul', 'runtime(s)'))
        start = time.time()
        log = {'iteration': [], 'beta': [], 'loss_a': [], 'loss_b': [], 'loss_m': [], 'loss': [], 'loss_mul': [],\
                'L1-error': [], 'objective': [], 'objective-error': [], 'runtime': []}
        epsilon = 1.
        low, high = [0., tb[0]-epsilon], [0., tb[1]+epsilon]
        r, t = domain_sampler(n_sample, low, high)
        b = tf.constant(1, dtype=DTYPE)
        epoch, tau = 0, 10
        while epoch < epochs+1:
            for _ in range(tau):
                loss_a, loss_b, loss_m, L, b = self.train_step(r, t, b)
            for _ in range(tau):
                loss_mul = self.train_step_mul(t, b)
            if epoch % 10 == 0:
                step_details = [epoch, loss_a.numpy(), loss_b.numpy(), loss_m.numpy(), L.numpy(), loss_mul.numpy(), time.time()-start]
                print('{:6d}{:15.6f}{:15.6f}{:10.4f}{:10.4f}{:15.4f}{:12.4f}'.format(*step_details))
                if epoch % 100 == 0:
                    error, objective, objective_error = self.error()
                    log['iteration'].append(step_details[0])
                    log['loss_a'].append(step_details[1])
                    log['loss_b'].append(step_details[2])
                    log['loss_m'].append(step_details[3])
                    log['loss'].append(step_details[4])
                    log['loss_mul'].append(step_details[5])
                    log['L1-error'].append(error)
                    log['objective'].append(objective)
                    log['objective-error'].append(objective_error)
                    log['beta'].append(b.numpy())
                    log['runtime'].append(step_details[6])
                    self.net.save_weights('{}/{}_{}'.format(self.save_folder, self.net.name, epoch))
            if b.numpy() < 2000.:
                b+=1
            epoch += 2*tau
        pd.DataFrame(log).to_csv('{}/train_log.csv'.format(self.save_folder), index=None)
        self.net.save_weights('{}/{}'.format(self.save_folder, self.net.name))

            
save_folder = '../data/helicoid-al'
Solver(save_folder=save_folder).learn(epochs=50000, n_sample=int(1e4))


#  extra test code
# r, t = domain_sampler(n_sample, low, high)
# if loss_a.numpy() > 0.5*b*loss_b.numpy():
#     b += (4.*loss_a.numpy() / loss_b.numpy()) + 1
# else:
#     b -= (4.*loss_a.numpy() / loss_b.numpy()) + 1
# if abs((loss_a-l).numpy()) < 1e-5 and loss_b.numpy() < 1e-5:
#         pass