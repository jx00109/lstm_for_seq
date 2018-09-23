# -*- coding: utf-8 -*-
import tensorflow as tf


def multi_layer_lstm(input_x, n_hidden, n_layers):
    '''
    返回静态多层LSTM单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_dims]
        n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
        n_layers: LSTM的层数
    '''

    # 可以看做3个隐藏层
    stacked_rnn = []
    for i in range(n_layers):
        stacked_rnn.append(tf.contrib.rnn.LSTMCell(num_units=n_hidden))

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层
    mcell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

    # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_dim)大小的张量
    hiddens, states = tf.nn.dynamic_rnn(mcell, input_x, time_major=False, dtype=tf.float32)

    return hiddens, states
