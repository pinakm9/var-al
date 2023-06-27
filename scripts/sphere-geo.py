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

root2 = np.sqrt(2., dtype=DTYPE)
root3 = np.sqrt(3., dtype=DTYPE)


v1 = np.array([-1., -1., 1 ], dtype=DTYPE)
v1 /= np.linalg.norm(v1) #/ root3
z, x = 0.0, 0.8
v2 = np.array([x, -np.sqrt((1-z**2-x**2)), z], dtype=DTYPE) 
w = v2 - np.dot(v1, v2)*v1
w = w/np.linalg.norm(w, ord=2)

alpha = np.arccos(v1[2], dtype=DTYPE)
phi_0 = np.arctan2(v1[1], v1[0], dtype=DTYPE) 
if phi_0 < 0:
    phi_0 += 2.*np.pi
beta = np.arccos(v2[2], dtype=DTYPE)
phi_1 = np.arctan2(v2[1], v2[0], dtype=DTYPE)
if phi_1 < 0:
    phi_1 += 2.*np.pi
gamma = np.arccos(np.dot(v1, v2), dtype=DTYPE)

def keep_neg(x):
    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return 1-out

def true(t):
    t1 = gamma * (t - alpha) / (beta - alpha)
    y =  tf.cos(t1) * v1[1] + tf.sin(t1) * w[1]
    x =  tf.cos(t1) * v1[0] + tf.sin(t1) * w[0]
    phi = tf.math.atan2(y, x)
    return phi + 2.*np.pi * keep_neg(phi)


print(gamma)
# exit()
tb = [alpha, beta]
objective = gamma


def boundary(u):
    o = tf.ones(shape=(1, 1), dtype=DTYPE)
    return tf.sqrt((u(o*alpha) - o*phi_0)**2 + (u(o*beta) - o*phi_1)**2)


def domain_sampler(n_sample):
    t = tf.random.uniform(shape=(n_sample, 1), minval=tb[0], maxval=tb[1], dtype=DTYPE)
    return t


gl = it.Gauss_Legendre(domain = [tb[0], tb[1]], num=100, dtype=DTYPE, d=2)
t0, w0 = tf.convert_to_tensor(gl.nodes.reshape(-1, 1), dtype=DTYPE), tf.convert_to_tensor(gl.weights.reshape(-1, 1), dtype=DTYPE)

@tf.function
def length(u):
    with tf.GradientTape() as tape:
        tape.watch(t0)
        u_ = u(t0)
    u_t = tape.gradient(u_, t0)
    return tf.reduce_sum(tf.sqrt(1.0 + (tf.sin(t0)*u_t)**2) * w0)

@tf.function
def length_(u, t):
    with tf.GradientTape() as tape:
        tape.watch(t)
        u_ = u(t)
    u_t = tape.gradient(u_, t)
    return tf.reduce_mean(tf.sqrt(1.0 + (tf.sin(t)*u_t)**2)) * (beta-alpha)

# print(objective)
# print(length(true))# == 0
# exit()


@tf.function
def L2_error(u):
    z = tf.zeros(shape=(1, 1), dtype=DTYPE)
    ft = true(t0)
    f = u(t0)
    return tf.sqrt( tf.reduce_sum((f - ft)**2 * w0) / tf.reduce_sum(w0) )



# @tf.function
def constraint_error(u):
    return boundary(u).numpy()[0][0]

"""
Things to track: 1) length or main loss, 2) boundary loss, 3) runtime, 4) iteration, 5) save every 100 steps
                 6) total loss
"""

class Solver:

    def __init__(self, save_folder, model_path=None) -> None:
        self.net =  arch.VanillaNet(50, 3, DTYPE, name='sphere-geodesic')
        self.save_folder = save_folder
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()
     
    def error(self):
        e = L2_error(self.net)
        o = length(self.net)
        ce = constraint_error(self.net)
        return e.numpy(), o.numpy(), (o/objective-1.).numpy(), ce

    @tf.function
    def train_step(self, t, b, g):
        with tf.GradientTape() as tape:
            loss_a = length(self.net) * g
            loss_b = tf.reduce_mean(boundary(self.net)**2)
            L = loss_a + 0.5*b*loss_b
        grads = tape.gradient(L, self.net.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss_a, loss_b, L

    def learn(self, epochs=10000, n_sample=1000):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=epochs, decay_rate=1e-3, staircase=False)
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        print("{:>6}{:>12}{:>12}{:>12}{:>18}".format('iteration', 'loss_a', 'loss_b', 'loss', 'runtime(s)'))
        start = time.time()
        log = {'iteration': [], 'beta': [], 'loss_a': [], 'loss_b': [], 'loss': [],  'L2-error': [],\
              'objective': [], 'objective-error': [], 'constraint-error': [], 'runtime': []}
        b = tf.constant(1., dtype=DTYPE)
        g = b/b
        epoch, tau = 0, 1
        t = domain_sampler(n_sample)
        while epoch < epochs+1:
            loss_a, loss_b, L = self.train_step(t, b, g)
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
                t = domain_sampler(n_sample)

            if epoch == 1000:
                a_, b_ = loss_a.numpy(), 0.5*b.numpy()*loss_b.numpy()
                c_ = np.ceil(np.log10(b_/a_))
                g = np.float32(10**c_) * tf.ones_like(b) 
        pd.DataFrame(log).to_csv('{}/train_log.csv'.format(self.save_folder), index=None)
        self.net.save_weights('{}/{}'.format(self.save_folder, self.net.name))

            
save_folder = '../data/sphere-geodesic'
Solver(save_folder=save_folder).learn(epochs=50000, n_sample=int(1e3))