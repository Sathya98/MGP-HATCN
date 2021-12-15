import os
import sys
import tensorflow as tf
import numpy as np

# appending head path
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)

# heavy keras imports
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, Add, Concatenate, Conv1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras import backend as K


class Attention(layers.Layer):
    '''
    Custom Attention Layer
    '''

    def __init__(self, output_size, expand=True):
        super(Attention, self).__init__()
        self.dense = tf.keras.layers.Dense(units = output_size)
        self.expand = expand

    def call(self, inp):
        '''
        The mathematical equation that we are trying to implement here is as follows:

        alpha = softmax(tanh(input*W)) , where alpha is the attention weight with shape = (timesteps, 1)
        gamma = Relu(transpose(input)*alpha)

        :param input: output of the convolution activation layer, shape = (timesteps, number of kernel filters)
        :return: within layer attention output (gamma), shape = (number of kernel filters, 1)
        '''
        
        input_for_alpha = tf.squeeze(tf.nn.tanh(self.dense(inp)))
        alpha = tf.expand_dims(tf.nn.softmax(input_for_alpha),axis=-1)
        gamma = tf.nn.relu(tf.matmul(inp, alpha, transpose_a=True))
        # print(f'attn call input: {inp.shape}, input_for_alpha: {input_for_alpha.shape}, alpha: {alpha.shape}, gamma: {gamma.shape}')
        if self.expand:
            return tf.expand_dims(tf.squeeze(gamma),axis=1)
        else:
            return tf.squeeze(gamma)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1], 1)
    
    

class HATCN:
    def __init__(self,
                 time_window,
                 n_channels,
                 num_layers,
                 DO,
                 L2reg,
                 kernel_size=2,
                 stride=1,
                 residual=True):
        
        
        att_outs = []
        history_seq = Input(shape=(time_window, n_channels))
        residual_input = history_seq
        
        for i in range(num_layers):  # num_layers is a tunable parameter.

            z=Conv1D(filters=n_channels, kernel_size=kernel_size, strides=stride, padding='causal',
                                      dilation_rate = 2**i, activation=tf.nn.relu,
                                      kernel_regularizer=l2(L2reg[i]),
                                      name="conv{}0".format(i))(residual_input)
            z=Dropout(DO[i],name='DO0'+str(i))(z)

            z=Conv1D(filters=n_channels, kernel_size=kernel_size, strides=stride, padding='causal',
                                      dilation_rate = 2**i, activation=tf.nn.relu,
                                      kernel_regularizer=l2(L2reg[i]),
                                      name="conv{}1".format(i))(z)
            z=Dropout(DO[i],name='DO1'+str(i))(z)
            
            att = Attention(1)(z)
            
            att_outs.append(att)
            
            if residual:
                residual_input=tf.nn.relu(Add()([residual_input,z]))
            else:
                residual_input=z
        
        int_attention = tf.concat(att_outs,axis=1)
        final_att = Attention(1,expand=False)(int_attention)
        output = Dense(2,input_shape=[n_channels])(final_att)

        self.model = Model(history_seq, output)

        self.trainable_variables = self.model.trainable_variables

    def __call__(self, inputs):
        # Note that the activation on alpha and the output are only valid if for a model trained on the last timestep
        return self.model(inputs)

    def get_weights(self):
        self.trainable_variables = self.model.trainable_variables
        return self.trainable_variables
    
    def set_weights(self, weights):
        if not isinstance(weights[0], np.ndarray):
            weights = [weights[i].numpy() for i in range(len(weights))]
        self.model.set_weights(weights)
