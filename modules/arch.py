import tensorflow as tf

class LSTMForgetBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes, dtype=tf.float32):
        super().__init__(name='LSTMForgetBlock', dtype=dtype)
        self.W_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_f', use_bias=False)
        self.U_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_f')
        self.W_g = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_g', use_bias=False)
        self.U_g = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_g')
        self.W_r = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_r', use_bias=False)
        self.U_r = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_r')
        self.W_s = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_s', use_bias=False)
        self.U_s = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_s')

    def call(self, x, h, c):
        f = tf.keras.activations.tanh(self.W_f(x) + self.U_f(h))
        g = tf.keras.activations.tanh(self.W_g(x) + self.U_g(h))
        r = tf.keras.activations.tanh(self.W_r(x) + self.U_r(h))
        s = tf.keras.activations.tanh(self.W_s(x) + self.U_s(h))
        c = f*c + g*s
        return r*tf.keras.activations.tanh(c), c


class LSTMForgetNet(tf.keras.models.Model):
    """
    Description: 
        LSTM Forget architecture
    Args:
        num_nodes: number of nodes in each LSTM layer
        num_layers: number of LSTM layers
    """
    def __init__(self, num_nodes, num_blocks, dtype=tf.float32, name = 'LSTMForgetNet'):
        super().__init__(dtype=dtype, name=name)
        self.num_nodes = num_nodes
        self.num_blocks = num_blocks
        self.lstm_blocks = [LSTMForgetBlock(num_nodes, dtype=dtype) for _ in range(num_blocks)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=None, dtype=dtype)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)
        
        

    def call(self, *args):
        x = tf.concat(args, axis=1)
        h = tf.zeros_like(x)
        c = tf.zeros((x.shape[0], self.num_nodes), dtype=self.dtype)
        for i in range(self.num_blocks):
            h, c = self.lstm_blocks[i](x, h, c)
            # h = self.batch_norm(h)
            # c = self.batch_norm(c)
        y = self.final_dense(h)
        return y

class VanillaNet(tf.keras.models.Model):

    def __init__(self, num_nodes, num_layers, dtype=tf.float32, name = 'VanillaNet'):
        super().__init__(dtype=dtype, name=name)
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.dense_layers = [tf.keras.layers.Dense(units=num_nodes, activation=tf.keras.activations.tanh) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=1, activation=None, dtype=dtype)
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=1)

    def call(self, *args):
        x = tf.concat(args, axis=1)
        for i in range(self.num_layers):
            x = self.dense_layers[i](x)
            x = self.batch_norm(x)
        y = self.final_dense(x)
        return y