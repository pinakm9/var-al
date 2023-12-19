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

# seed = 42
# np.random.seed(seed)
# tf.random.set_seed(seed)

DTYPE = 'float32'


# points =========< start

theta_0, phi_0 = np.pi/4, np.pi/4
X0 = np.array([np.sin(theta_0)*np.cos(phi_0), np.sin(theta_0)*np.sin(phi_0), np.cos(theta_0)], dtype=DTYPE) 

theta_1, phi_1 = np.pi/2., 3.*np.pi/4
X1 = np.array([np.sin(theta_1)*np.cos(phi_1), np.sin(theta_1)*np.sin(phi_1), np.cos(theta_1)], dtype=DTYPE)


X2 = X1 - np.dot(X0, X1)*X0
X2 /= np.sqrt(1. - np.dot(X0, X1)**2)

objective = np.arccos(np.dot(X0, X1))
print(objective)

dtheta = theta_1 - theta_0
tan0, tan1 = np.tan(theta_0), np.tan(theta_1)
sindt, tandt = np.sin(dtheta), np.tan(dtheta)
b = phi_1 + np.arctan(1./tandt - tan1/(tan0 * sindt))
a = 1./(tan1 * np.cos(phi_1 - b))

# points end =========< end


# true great circle ==========< start

def keep_neg(x):
    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return 1-out

def true_phi(t):
    return np.arccos(1./(a * np.tan(t))) + b
#65 true great circle ============< end

# exit()
tb = [theta_0, theta_1]


def boundary(u, t):
    # o = tf.ones(shape=(1, 1), dtype=DTYPE)
    o = tf.ones_like(t)
    return tf.reduce_mean((u(theta_0*o) - phi_0*o)**2 + (u(o*theta_1) - o*phi_1)**2)


def domain_sampler(n_sample):
    t = tf.random.uniform(shape=(n_sample, 1), minval=tb[0], maxval=tb[1], dtype=DTYPE)
    return t


gl = it.Gauss_Legendre(domain = [tb[0], tb[1]], num=100, d=4, dtype=DTYPE)
t0, w0 = tf.convert_to_tensor(gl.nodes.reshape(-1, 1), dtype=DTYPE), tf.convert_to_tensor(gl.weights.reshape(-1, 1), dtype=DTYPE)

@tf.function
def length(u):
    with tf.GradientTape() as tape:
        tape.watch(t0)
        u_ = u(t0)
    u_t = tape.gradient(u_, t0)
    return tf.reduce_sum(tf.sqrt(1.0 + (tf.sin(t0)*u_t)**2) * w0)




@tf.function
def L2_error(u):
    z = tf.zeros(shape=(1, 1), dtype=DTYPE)
    ft = true_phi(t0)
    f = u(t0)
    return tf.sqrt(tf.reduce_sum((f - ft)**2 * w0) / tf.reduce_sum(w0) )



# @tf.function
def constraint_error(u):
    return tf.sqrt(boundary(u, t0)).numpy()

"""
Things to track: 1) length or main loss, 2) boundary loss, 3) runtime, 4) iteration, 5) save every 100 steps
                 6) total loss
"""



class LSTMForgetNet(tf.keras.models.Model):
    """
    Description: 
        LSTM Forget architecture
    Args:
        num_nodes: number of nodes in each LSTM layer
        num_layers: number of LSTM layers
    """
    def __init__(self, num_nodes, num_blocks, dtype=tf.float32, name = 'LSTMForgetNet', dim=1):
        super().__init__(dtype=dtype, name=name)
        self.num_nodes = num_nodes
        self.num_blocks = num_blocks
        self.lstm_blocks = [arch.LSTMForgetBlock(num_nodes, dtype=dtype) for _ in range(num_blocks)]
        self.final_dense = tf.keras.layers.Dense(units=dim, activation=tf.keras.activations.tanh, dtype=dtype)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
        
        

    def call(self, *args):
        x = tf.concat(args, axis=1)
        h = tf.zeros_like(x)
        c = tf.zeros((x.shape[0], self.num_nodes), dtype=self.dtype)
        for i in range(self.num_blocks):
            h, c = self.lstm_blocks[i](x, h, c)
            # h = self.batch_norm(h)
            # c = self.batch_norm(c)
        y = (self.final_dense(h)+1) * np.pi
        return y
    

class VanillaNet(tf.keras.models.Model):

    def __init__(self, num_nodes, num_layers, dtype=tf.float32, name='VanillaNet', dim=1):
        super().__init__(dtype=dtype, name=name)
        self.num_nodes = num_nodes
        self.num_layers = num_layers 
        if isinstance(num_nodes, list):
            self.dense_layers = [tf.keras.layers.Dense(units=num_nodes[i], activation=tf.keras.activations.tanh) for i in range(num_layers)]
        else:
            self.dense_layers = [tf.keras.layers.Dense(units=num_nodes, activation=tf.keras.activations.tanh) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=dim, activation=tf.keras.activations.tanh, dtype=dtype)
        self.batch_norms = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

    def call(self, *args):
        x = tf.concat(args, axis=1)
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
            #x = self.batch_norms[i](x)
        y =  (self.final_dense(x)+1) * np.pi
        return y
    

class Solver:

    def __init__(self, save_folder, model_path=None) -> None:
        self.net =  LSTMForgetNet(50, 3, DTYPE, name='sphere-geodesic')
        self.save_folder = save_folder
        if model_path is not None:
            self.net.load_weights(model_path).expect_partial()
     
    def error(self):
        e = L2_error(self.net)
        o = length(self.net)
        ce = constraint_error(self.net)
        return e.numpy(), o.numpy(), (o/objective-1.).numpy(), ce

    @tf.function
    def train_step(self, t, b):
        with tf.GradientTape() as tape:
            loss_a = length(self.net) 
            loss_b = boundary(self.net, t)
            L = loss_a + 0.5*b*loss_b
        grads = tape.gradient(L, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss_a, loss_b, L

    def learn(self, epochs=10000, n_sample=1000):
        print("{:>6}{:>12}{:>12}{:>12}{:>18}".format('iteration', 'loss_a', 'loss_b', 'loss', 'runtime(s)'))
        start = time.time()
        log = {'iteration': [], 'theta_1': [], 'loss_a': [], 'loss_b': [], 'loss': [],  'L2-error': [],\
              'objective': [], 'objective-error': [], 'constraint-error': [], 'runtime': []}
        epoch, tau, b0, delb, maxb = 0, 10, 100, 1.01, 500
        initial_rate = 1e-3
        decay_rate = 1e-1
        decay_steps = int(2*tau)
        final_learning_rate = 1e-4
        final_decay_rate = 1e-2
        drop = 1.0
        tipping_point = int(2*tau*(maxb-b0)/delb)
        final_decay_steps = epochs - tipping_point
        lr_schedule = arch.CyclicLR(initial_rate, decay_rate, decay_steps,final_learning_rate, final_decay_rate, final_decay_steps,\
                            drop, tipping_point)
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=epochs, decay_rate=1e-1, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        b = tf.constant(b0, dtype=DTYPE)
        g = b/b
        t = domain_sampler(n_sample)
        past_a, tol = g*1., 1e-2
        while epoch < epochs+1:
            loss_a, loss_b, L = self.train_step(t, b)
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
                    log['theta_1'].append(b.numpy())
                    log['runtime'].append(step_details[4])
                    self.net.save_weights('{}/{}_{}'.format(self.save_folder, self.net.name, epoch))
                
            epoch += 1

            if epoch % tau == 0:
                if b < maxb:
                    b *= delb
                t = domain_sampler(n_sample)

        pd.DataFrame(log).to_csv('{}/train_log.csv'.format(self.save_folder), index=None)
        self.net.save_weights('{}/{}'.format(self.save_folder, self.net.name))

            
save_folder = '../data/sphere-geodesic'
Solver(save_folder=save_folder).learn(epochs=50000, n_sample=int(1e3))
np.save('{}/points.npy'.format(save_folder), np.vstack([X0, X1]))