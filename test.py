import unittest
import tensorflow as tf
import numpy as np
from model import *

def value_distance(val_a, val_b):
    return np.sum(np.abs(val_a - val_b)) / np.sum(np.abs(val_a))

class TestInference(unittest.TestCase):

    def test_dynamic_rnn(self):

        test_case = [[1., 0.], [2., 3.], [5., 4.]]

        single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(2, reuse=tf.get_variable_scope().reuse)
        lstm_fw_cell = single_cell()

        inputs = tf.placeholder(tf.float32, [None, None, 2])
        outputs, _ = tf.nn.dynamic_rnn(lstm_fw_cell, inputs, dtype=tf.float32, time_major=False)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        tf_vals = sess.run(outputs, feed_dict={inputs: np.array([test_case])})

        # reduce 1st dim (batch dimension)
        tf_vals = tf_vals[0]

        # retrieve kernel and bias from the model
        weights = sess.run(lstm_fw_cell.weights)
        fw = weights[0]
        fb = weights[1]

        np_vals, _ = dynamic_rnn(np.array(test_case), fw, fb)
        np_vals = np.array(np_vals)

        # for debug
        # print("tensorflow inference")
        # print(tf_vals)
        #
        # print("numpy inferece")
        # print(np_vals)

        assert tf_vals.shape == np_vals.shape

        assert value_distance(tf_vals, np_vals) < 0.001

    def test_bidirectional_lstm(self):

        test_case = [[1., 0.], [2., 3.], [5., 4.]]

        single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(2, reuse=tf.get_variable_scope().reuse)
        lstm_fw_cell = single_cell()
        lstm_bw_cell = single_cell()

        inputs = tf.placeholder(tf.float32, [None, None, 2])
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32, time_major=False)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf_vals = sess.run(outputs, feed_dict={inputs: np.array([test_case])})

        # reduce 1st dim (batch dimension)
        tf_fw_vals = tf_vals[0][0]
        tf_bw_vals = tf_vals[1][0]

        # export weights
        fwb = sess.run(lstm_fw_cell.weights)
        fw = fwb[0]
        fb = fwb[1]

        bwb = sess.run(lstm_bw_cell.weights)
        bw = bwb[0]
        bb = bwb[1]

        np_fw_vals, np_bw_vals = bidirectional_dynamic_rnn(np.array(test_case), fw, fb, bw, bb)

        # print("tensorflow inference")
        # print(tf_fw_vals)
        #
        # print("numpy inference")
        # print(np_fw_vals)

        assert tf_fw_vals.shape == np_fw_vals.shape
        assert tf_bw_vals.shape == np_bw_vals.shape

        assert value_distance(tf_fw_vals, np_fw_vals) < 0.001
        assert value_distance(tf_bw_vals, np_bw_vals) < 0.001

    def test_stack_bidirectional_lstm(self):

        layer_size = 2

        test_case = [[1., 0.], [2., 3.], [5., 4.]]

        single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(2, reuse=tf.get_variable_scope().reuse)
        lstm_fw_cell = [single_cell() for _ in range(layer_size)]
        lstm_bw_cell = [single_cell() for _ in range(layer_size)]

        inputs = tf.placeholder(tf.float32, [None, None, 2])
        (outputs, output_state_fw, output_state_bw) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                                                     lstm_bw_cell,
                                                                                                     inputs,
                                                                                                     dtype=tf.float32,
                                                                                                     time_major=False)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf_vals = sess.run(outputs, feed_dict={inputs: np.array([test_case])})

        # reduce dim
        tf_vals = tf_vals[0]

        fw_lst = []
        bw_lst = []
        fb_lst = []
        bb_lst = []

        for i in range(layer_size):
            wb = sess.run(lstm_fw_cell[i].weights)
            fw_lst.append(wb[0])
            fb_lst.append(wb[1])

            wb = sess.run(lstm_bw_cell[i].weights)
            bw_lst.append(wb[0])
            bb_lst.append(wb[1])

        np_vals = stack_bidirectional_dynamic_rnn(np.array(test_case), fw_lst, fb_lst, bw_lst, bb_lst)

        #print("tensorflow inference")
        #print(tf_vals)
        #rint("numpy inference")
        #print(np_vals)

        assert tf_vals.shape == np_vals.shape
        assert value_distance(tf_vals, np_vals) < 0.001

if __name__ == '__main__':
    unittest.main()