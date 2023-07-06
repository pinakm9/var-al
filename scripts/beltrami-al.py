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
import dom
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


DTYPE = 'float64'
box = dom.Box3D(dtype=DTYPE)

def true(x, y, z):
    p = tf.sin(z) + tf.cos(y)
    q = tf.sin(x) + tf.cos(z)
    r = tf.sin(y) + tf.cos(x)
    return tf.concat([p, q, r], axis=-1)

def boundary_sampler(n_sample):
    x, y, z = [], [], []
    for side in box.boundary_sample(int(np.ceil(n_sample/6))):
        x.append(side[0])     
        y.append(side[1]) 
        z.append(side[2]) 
    return tf.concat(x, axis=0), tf.concat(y, axis=0), tf.concat(z, axis=0)

def domain_sampler(n_sample):
    return box.sample(n_sample)

def boundary(u, x, y, z):
    return u(x, y, z) - true(x, y, z)

@tf.function
def energy(u, x, y, z):
    return 0.5 * tf.reduce_mean(tf.reduce_sum(u(x, y, z)**2, axis=-1, keepdims=True)) 

objective = 1.5 

x0, y0, z0 = domain_sampler(int(2e4))

@tf.function
def energy_0(u):
    return 0.5 * tf.reduce_mean(tf.reduce_sum(u(x0, y0, z0)**2, axis=-1, keepdims=True))


@tf.function
def L2_error(u, x, y, z):
    f = tf.reduce_sum((u(x, y, z) - true(x, y, z))**2, axis=-1, keepdims=True)  
    return tf.sqrt(tf.reduce_mean(f))

xb, yb, zb = boundary_sampler(int(1e4))

@tf.function
def constraint_error(u, x, y, z):
    f = tf.reduce_sum(tf.square(boundary(u, x, y, z)), axis=-1, keepdims=True)
    return tf.sqrt(tf.reduce_mean(f))

def curl(f, x, y, z):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y, z])
        Ax, Ay, Az = tf.split(f(x, y, z), 3, axis=-1)
    Ax_y = tape.gradient(Ax, y)
    Ay_x = tape.gradient(Ay, x)
    Ax_z = tape.gradient(Ax, z)
    Az_x = tape.gradient(Az, x)
    Ay_z = tape.gradient(Ay, z)
    Az_y = tape.gradient(Az, y)
    return tf.concat([(Az_y - Ay_z), (Ax_z - Az_x), (Ay_x - Ax_y)], axis=-1) 



class Solver:

    def __init__(self, save_folder, model_path=None) -> None:
        self.net =  arch.LSTMForgetNet(50, 3, DTYPE, dim=3, name='Beltrami-al')
        self.mul = arch.VanillaNet(50, 3, DTYPE, dim=3, name='Beltrami-al-mul')
        self.save_folder = save_folder
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()

    @tf.function
    def B(self, x, y, z): 
        return curl(self.net, x, y, z)
    

    def error(self, x, y, z, x_, y_, z_):
        e = L2_error(self.B, x, y, z)
        o = energy_0(self.B)
        ce = constraint_error(self.B, x_, y_, z_)
        return e.numpy(), o.numpy(), (o/objective-1.).numpy(), ce.numpy()

    @tf.function
    def train_step(self, x, y, z, x_, y_, z_, b):
        with tf.GradientTape() as tape:
            loss_a = energy(self.B, x, y, z) 
            loss_b = tf.reduce_mean(tf.reduce_sum(tf.square(boundary(self.net, x_, y_, z_)), axis=-1, keepdims=True))
            loss_m = tf.reduce_mean(tf.reduce_sum(boundary(self.net, x_, y_, z_) * self.mul(x_, y_, z_), axis=-1, keepdims=True))
            L = loss_a + 0.5*b*loss_b + loss_m
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss_a, loss_b, loss_m, L


    @tf.function
    def train_step_mul(self, mu_0,  x_, y_, z_, b):
        with tf.GradientTape() as tape:
            L = tf.reduce_mean(tf.reduce_sum(tf.square(self.mul(x_, y_, z_) - mu_0 - b * boundary(self.net,  x_, y_, z_)), axis=-1, keepdims=True))
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
        decay_rate = 1e-1
        decay_steps = int(2*tau)
        final_learning_rate = 1e-5
        final_decay_rate = 1e-2
        drop = 1.0
        tipping_point = int(2*tau*(maxb-b0)/delb)
        final_decay_steps = epochs - tipping_point
        lr_schedule = arch.CyclicLR(initial_rate, decay_rate, decay_steps,final_learning_rate, final_decay_rate, final_decay_steps,\
                            drop, tipping_point)
        lr_schedule_mul = arch.CyclicLR(initial_rate, decay_rate, decay_steps,final_learning_rate, final_decay_rate, final_decay_steps,\
                            drop, tipping_point)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.optimizer_mul = tf.keras.optimizers.Adam(learning_rate=lr_schedule_mul)
        b = tf.constant(b0, dtype=DTYPE)
        x, y, z = domain_sampler(n_sample)
        x_, y_, z_ = boundary_sampler(n_sample)
        while epoch < epochs+1:
            for _ in range(tau):
                loss_a, loss_b, loss_m, L = self.train_step(x, y, z, x_, y_, z_, b)
            mu_0 = self.mul(x_, y_, z_)
            for _ in range(tau):
                loss_mul = self.train_step_mul(mu_0, x_, y_, z_, b)
            if epoch % 10 == 0:
                step_details = [epoch, loss_a.numpy(), loss_b.numpy(), loss_m.numpy(), L.numpy(), loss_mul.numpy(), time.time()-start]
                print('{:6d}{:15.6f}{:15.6f}{:10.4f}{:10.4f}{:15.4f}{:12.4f}'.format(*step_details))
                if epoch % 100 == 0:
                    error, objective, objective_error, ce = self.error(x, y, z, x_, y_, z_)
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
            x, y, z = domain_sampler(n_sample)
            x_, y_, z_ = boundary_sampler(n_sample)

  
                
        pd.DataFrame(log).to_csv('{}/train_log.csv'.format(self.save_folder), index=None)
        self.net.save_weights('{}/{}'.format(self.save_folder, self.net.name))

            
save_folder = '../data/Beltrami-al'
Solver(save_folder=save_folder).learn(epochs=50000, n_sample=int(1e3))


