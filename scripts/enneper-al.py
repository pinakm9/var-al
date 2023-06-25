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
import integrator as it
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


DTYPE = 'float32'
R = .5
rb = [0., R]
tb = [-np.pi, np.pi]


def Enneper_boundary(u, t):
    r = R*tf.ones_like(t)
    x = r*tf.cos(t) - r*r*r*tf.cos(3.*t)/3.
    y = -r*tf.sin(t) - r*r*r*tf.sin(3.*t)/3.
    r = tf.sqrt(x*x+y*y)
    t = tf.math.atan2(y, x)
    return u(r, t) - r*r*tf.cos(2.*t)

def domain_sampler(n_sample, low=[rb[0], tb[0]], high=[rb[1], tb[1]]):
    r = tf.random.uniform(shape=(n_sample, 1), minval=low[0], maxval=high[0], dtype=DTYPE)
    t = tf.random.uniform(shape=(n_sample, 1), minval=low[1], maxval=high[1], dtype=DTYPE)
    return r, t


gl = it.Gauss_Legendre_2D(domain = [[rb[0], tb[0]], [rb[1], tb[1]]], num=13, dtype=DTYPE, d=7)
r0, t0, w0 = tf.convert_to_tensor(gl.x, dtype=DTYPE), tf.convert_to_tensor(gl.y, dtype=DTYPE), tf.convert_to_tensor(gl.w, dtype=DTYPE)

gl1 = it.Gauss_Legendre_2D(domain = [[rb[0], tb[0]], [rb[1], tb[1]]], num=15, dtype=DTYPE, d=10)
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
    return tf.concat([r*tf.cos(t) - r*r*r*tf.cos(3.*t)/3., -r*tf.sin(t) - r*r*r*tf.sin(3.*t)/3., r*r*tf.cos(2.*t)], axis=-1)

objective = area1(true)

c1 = tf.reduce_sum(tf.ones_like(w1) * w1)

@tf.function
def L1_error(u):
    f = tf.sqrt(tf.reduce_sum((u(r1, t1) - true(r1, t1))**2, axis=-1, keepdims=True)) 
    return tf.reduce_sum(f * w1) / c1
"""
Things to track: 1) area or main loss, 2) boundary loss, 3) runtime, 4) iteration, 5) save every 100 steps
                 6) total loss
"""

gl2 = it.Gauss_Legendre(domain = [tb[0], tb[1]], num=15, dtype=DTYPE, d=10)
t2 = tf.convert_to_tensor(gl2.nodes.reshape(-1, 1), dtype=DTYPE)
w2 = tf.convert_to_tensor(gl2.weights.reshape(-1, 1), dtype=DTYPE)
c2 = tf.reduce_sum(tf.ones_like(w2) * w2)

@tf.function
def constraint_error(u):
    f = tf.abs(Enneper_boundary(u, t2))
    return tf.reduce_sum(f * w2) / c2

class Solver:

    def __init__(self, save_folder, model_path=None) -> None:
        self.net = arch.LSTMForgetNet(50, 3, DTYPE, name='Enneper-al')
        self.mul = arch.VanillaNet(64, 3, DTYPE, name='Enneper-mul-al')
        self.save_folder = save_folder
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()
        self.r0, self.t0 = domain_sampler(100000)

    def error(self):
        e = L1_error(self.net)
        o = area1(self.net)
        ce = constraint_error(self.net)
        return e.numpy(), o.numpy(), (o/objective - 1.).numpy(), ce.numpy()

    @tf.function
    def train_step(self, r, t, b):
        with tf.GradientTape() as tape:
            loss_a = area(self.net) * 1e-2
            loss_b = tf.reduce_mean(Enneper_boundary(self.net, t)**2)
            loss_m = tf.reduce_mean(Enneper_boundary(self.net, t) * self.mul(t))
            L = loss_a + 0.5*b*loss_b + loss_m
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss_a, loss_b, loss_m, L, b


    @tf.function
    def train_step_mul(self, t, b):
        mu_0 = self.mul(t)
        with tf.GradientTape() as tape:
            L = tf.reduce_mean((self.mul(t) - mu_0 - b * Enneper_boundary(self.net, t))**2)
        grads = tape.gradient(L, self.mul.trainable_weights)
        self.optimizer_mul.apply_gradients(zip(grads, self.mul.trainable_weights))
        return L

    def learn(self, epochs=10000, n_sample=1000):
        ir, dr = 1e-7, 1e-1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(ir, decay_steps=epochs, decay_rate=dr, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        lr_schedule_mul = tf.keras.optimizers.schedules.ExponentialDecay(ir, decay_steps=epochs, decay_rate=dr, staircase=False)
        self.optimizer_mul = tf.keras.optimizers.Adam(learning_rate=lr_schedule_mul)
        print("{:>6}{:>12}{:>12}{:>12}{:>8}{:>12}{:>18}"\
              .format('iteration', 'loss_a', 'loss_b', 'loss_m', 'loss', 'loss_mul', 'runtime(s)'))
        start = time.time()
        log = {'iteration': [], 'beta': [], 'loss_a': [], 'loss_b': [], 'loss_m': [], 'loss': [], 'loss_mul': [],\
                'L1-error': [], 'objective': [], 'objective-error': [], 'constraint-error': [], 'runtime': []}
        epsilon = 0.
        low, high = [rb[0], tb[0]], [rb[1], tb[1]]
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
                    error, objective, objective_error, ce = self.error()
                    log['iteration'].append(step_details[0])
                    log['loss_a'].append(step_details[1])
                    log['loss_b'].append(step_details[2])
                    log['loss_m'].append(step_details[3])
                    log['loss'].append(step_details[4])
                    log['loss_mul'].append(step_details[5])
                    log['L1-error'].append(error)
                    log['objective'].append(objective)
                    log['objective-error'].append(objective_error)
                    log['constraint-error'].append(ce)
                    log['beta'].append(b.numpy())
                    log['runtime'].append(step_details[6])
                    self.net.save_weights('{}/{}_{}'.format(self.save_folder, self.net.name, epoch))
            if b.numpy() < 1e3:
                        b += 1.
            r, t = domain_sampler(n_sample, low, high)
            epoch += 2*tau
        pd.DataFrame(log).to_csv('{}/train_log.csv'.format(self.save_folder), index=None)
        self.net.save_weights('{}/{}'.format(self.save_folder, self.net.name))

            
save_folder = '../data/Enneper-al'
Solver(save_folder=save_folder).learn(epochs=50000, n_sample=int(1e3))
