import tensorflow as tf
import numpy as np
from model import multi_layer_lstm

N_STEPS = 240  # 时序信号的步长
N_DIMS = 30  # 时序信号的特征维度
N_TARGET = 24  # 预测值的维度
N_HIDDENS = 128  # lstm隐藏层的维度
N_LAYERS = 3  # lstm的层数
LR = 0.001  # 学习率
TRAIN_STEPS = 1  # 训练迭代次数
BATCH_SIZE = 2  # 每次迭代用的时序信号个数
DISPALY_STEPS = 1  # 显示的频率


def run_lstm():
    input_x = tf.placeholder(dtype=tf.float32, shape=[None, N_STEPS, N_DIMS])
    input_y = tf.placeholder(dtype=tf.float32, shape=[None, N_TARGET])

    # hidden shape: [batch_size, n_steps, n_hidden]
    hiddens, states = multi_layer_lstm(input_x, N_HIDDENS, N_LAYERS)

    # 取LSTM最后一个时序的输出，然后经过全连接网络得到输出值
    output = tf.contrib.layers.fully_connected(inputs=hiddens[:, -1, :],
                                               num_outputs=N_TARGET,
                                               activation_fn=tf.nn.softmax)
    print(output.shape)
    # 代价函数 J =-(Σy.logaL)/n    .表示逐元素乘
    cost = tf.reduce_mean(-tf.reduce_sum(input_y * tf.log(output), axis=1))

    train = tf.train.AdamOptimizer(LR).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 开始迭代 使用Adam优化的随机梯度下降法
        for i in range(TRAIN_STEPS):
            x_batch = np.random.rand(BATCH_SIZE, N_STEPS, N_DIMS)
            y_batch = np.random.rand(BATCH_SIZE, N_TARGET)
            train.run(feed_dict={input_x: x_batch, input_y: y_batch})
            if (i + 1) % DISPALY_STEPS == 0:
                train_output, training_cost = sess.run([output, cost], feed_dict={input_x: x_batch, input_y: y_batch})
                print('Step {0}:Training set cost {1}.'.format(i + 1, training_cost))
                print(train_output)


if __name__ == "__main__":
    run_lstm()
