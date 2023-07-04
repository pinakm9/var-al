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


DTYPE  = 'float32'
R, a, b, c0 = 1., 1.2, -1., 1.1
rb, zb = [0.9*R, 1.1*R], [-0.1*R, 0.1*R]


def domain_sampler(n_sample, low=[rb[0], zb[0]], high=[rb[1], zb[1]]):
    r = tf.random.uniform(shape=(n_sample, 1), minval=low[0], maxval=high[0], dtype=DTYPE)
    z = tf.random.uniform(shape=(n_sample, 1), minval=low[1], maxval=high[1], dtype=DTYPE)
    return r, z


def pde(u, r, z):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([r, z])
        u_ = u(r, z)
        u_r, u_z  = tape.gradient(u_, [r, z])
    u_rr = tape.gradient(u_r, r)
    u_zz = tape.gradient(u_z, z)
    return tf.reduce_mean((u_zz + u_rr - u_r/r - a*r*r - b*R*R)**2)


def boundary_sampler(n_sample):
    r = tf.random.uniform(shape=(n_sample, 1), minval=rb[0], maxval=rb[1], dtype=DTYPE)
    z = tf.random.uniform(shape=(n_sample, 1), minval=zb[0], maxval=zb[1], dtype=DTYPE)
    r_r, z_r = rb[1] * tf.ones_like(z), z
    r_l, z_l = rb[0] * tf.ones_like(z), z
    r_u, z_u = r, zb[1] * tf.ones_like(r)
    r_d, z_d = r, zb[0] * tf.ones_like(r)
    return tf.concat([r_d, r_r, r_u, r_l], axis=0), tf.concat([z_d, z_r, z_u, z_l], axis=0) 


def true(r, z):
    r2, z2, R2 = r*r, z*z, R*R
    zeta = (r2 - R2) / (2.*R)
    A = 0.5 * (b + c0) * R2 * z2
    B = c0 * zeta * R * z2
    C = 0.5 * (a - c0) * R2 * zeta * zeta
    return A + B + C

def boundary(u, r, z):
    return u(r, z) - true(r, z)




gl = it.Gauss_Legendre_2D(domain = [[rb[0], zb[0]], [rb[1], zb[1]]], num=20, dtype=DTYPE, d=4)
r0, z0, w0 = tf.convert_to_tensor(gl.x, dtype=DTYPE), tf.convert_to_tensor(gl.y, dtype=DTYPE), tf.convert_to_tensor(gl.w, dtype=DTYPE)

def L2_error(u):
    f = (u(r0, z0) - true(r0, z0))**2 
    return tf.sqrt(tf.reduce_sum(f * w0) / tf.reduce_sum(w0))

gl = it.Gauss_Legendre(domain = [rb[0], rb[1]], num=20, dtype=DTYPE, d=4)
r1, wr1 = tf.convert_to_tensor(gl.nodes.reshape(-1, 1), dtype=DTYPE), tf.convert_to_tensor(gl.weights.reshape(-1, 1), dtype=DTYPE)

gl = it.Gauss_Legendre(domain = [zb[0], zb[1]], num=20, dtype=DTYPE, d=4)
z1, wz1 = tf.convert_to_tensor(gl.nodes.reshape(-1, 1), dtype=DTYPE), tf.convert_to_tensor(gl.weights.reshape(-1, 1), dtype=DTYPE)

one_r1, one_z1 = tf.ones_like(r1), tf.ones_like(z1)

# @tf.function
def constraint_error(u):
    down = (u(r1, zb[0]*one_r1) - true(r1, zb[0]*one_r1))**2 * wr1
    right = (u(rb[1]*one_z1, z1) - true(rb[1]*one_z1, z1))**2 * wz1
    up = (u(r1, zb[1]*one_r1) - true(r1, zb[1]*one_r1))**2 * wr1
    left = (u(rb[0]*one_z1, z1) - true(rb[0]*one_z1, z1))**2 * wz1
    length = 2. * (tf.reduce_sum(wr1) + tf.reduce_sum(wz1))
    return tf.sqrt((down + right + up + left) / length).numpy()[0][0]

@tf.function
def objective(u):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([r0, z0])
        u_ = u(r0, z0)
        u_r, u_z  = tape.gradient(u_, [r0, z0])
    u_rr = tape.gradient(u_r, r0)
    u_zz = tape.gradient(u_z, z0)
    return tf.sqrt(tf.reduce_sum(w0 * (u_zz + u_rr - u_r/r0 - a*r0*r0 - b*R*R)**2) / ((rb[1]-rb[0])*(zb[1]-zb[0])))

class Solver:

    def __init__(self, save_folder, model_path=None) -> None:
        self.net =  arch.LSTMForgetNet(50, 3, DTYPE, name='Grad-Shafranov')
        self.save_folder = save_folder
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()
    

    def error(self):
        e = L2_error(self.net)
        o = objective(self.net)
        ce = constraint_error(self.net)
        return e.numpy(), o.numpy(), ce

    @tf.function
    def train_step(self, r, z, r_, z_, b):
        with tf.GradientTape() as tape:
            loss_a = pde(self.net, r, z) 
            loss_b = tf.reduce_mean(boundary(self.net, r_, z_)**2)
            L = loss_a + 0.5*b*loss_b
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss_a, loss_b, L

    def learn(self, epochs=10000, n_sample=1000):
        print("{:>6}{:>12}{:>12}{:>12}{:>18}".format('iteration', 'loss_a', 'loss_b', 'loss', 'runtime(s)'))
        start = time.time()
        log = {'iteration': [], 'beta': [], 'loss_a': [], 'loss_b': [], 'loss': [],  'L2-error': [],\
               'objective-error': [], 'constraint-error': [], 'runtime': []}
        epoch, tau, b0, delb, maxb = 0, 10, 100, 1.01, 1000
        initial_rate = 1e-4
        decay_rate = 1e-1
        decay_steps = int(2*tau)
        final_learning_rate = 1e-6
        final_decay_rate = 1e-2
        drop = 1.0
        tipping_point = int(2*tau*(maxb-b0)/delb)
        final_decay_steps = epochs - tipping_point
        lr_schedule = arch.CyclicLR(initial_rate, decay_rate, decay_steps,final_learning_rate, final_decay_rate, final_decay_steps,\
                            drop, tipping_point)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        b = tf.constant(b0, dtype=DTYPE)
        r, z = domain_sampler(n_sample)
        r_, z_ = boundary_sampler(n_sample)
        
        while epoch < epochs+1:
            loss_a, loss_b, L = self.train_step(r, z, r_, z_, b)
            if epoch % 10 == 0:
                step_details = [epoch, loss_a.numpy(), loss_b.numpy(), L.numpy(), time.time()-start]
                print('{:6d}{:15.6f}{:15.6f}{:12.6f}{:12.4f}'.format(*step_details))
                if epoch % 100 == 0:
                    error, objective_error, ce = self.error()
                    log['iteration'].append(step_details[0])
                    log['loss_a'].append(step_details[1])
                    log['loss_b'].append(step_details[2])
                    log['loss'].append(step_details[3])
                    log['L2-error'].append(error)
                    # log['objective'].append(objective)
                    log['objective-error'].append(objective_error)
                    log['constraint-error'].append(ce)
                    log['beta'].append(b.numpy())
                    log['runtime'].append(step_details[4])
                    self.net.save_weights('{}/{}_{}'.format(self.save_folder, self.net.name, epoch))
                
            epoch += 1

            if epoch % 2*tau == 0:
                if b < maxb:
                    b *= delb
                r, z = domain_sampler(n_sample)
                r_, z_ = boundary_sampler(n_sample)

        pd.DataFrame(log).to_csv('{}/train_log.csv'.format(self.save_folder), index=None)
        self.net.save_weights('{}/{}'.format(self.save_folder, self.net.name))

            
save_folder = '../data/Grad-Shafranov'
Solver(save_folder=save_folder).learn(epochs=50000, n_sample=int(1e3))