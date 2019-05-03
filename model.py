import numpy as np

def lstm_cell(input_tensor, prev_hidden_tensor, prev_cell_state, kernel, bias):
    """
    forward inference logic of a lstm cell
    reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/lstm_ops.py

    :param input_tensor: input tensor
    :param prev_hidden_tensor: tensor of previous hidden state
    :param kernel: weight
    :param bias: bias
    :return: hidden tensor, cell state tensor
    """


    xh = np.concatenate([input_tensor, prev_hidden_tensor])
    h = np.dot(xh, kernel)+bias
    i, ci, f, o = np.split(h, 4)

    # embed sigmoid to reduce function call
    i = 1. / (1. + np.exp(-i))
    f = 1. / (1. + np.exp(-f))
    o = 1. / (1. + np.exp(-o))
    ci = np.tanh(ci)

    cs = np.multiply(ci, i) + np.multiply(prev_cell_state, f)
    co = np.tanh(cs)
    h = np.multiply(co, o)

    return h, cs

def dynamic_rnn(input_tensors, kernel, bias):
    """
    inference logic of an unidirectional lstm

    :param input_tensors: a list of input tensor
    :param kernel: weight
    :param bias: bias
    :return: hidden tensors, cell state tensors
    """

    hidden_size = int(bias.shape[0]/4)

    prev_hidden = np.zeros(hidden_size)
    prev_cell_state = np.zeros(hidden_size)

    hidden_lst = []
    cell_state_lst = []

    for input_tensor in input_tensors:
        hidden, cell_state = lstm_cell(input_tensor, prev_hidden, prev_cell_state, kernel, bias)
        hidden_lst.append(hidden)
        cell_state_lst.append(cell_state)
        prev_hidden = hidden
        prev_cell_state = cell_state

    return hidden_lst, cell_state_lst

def bidirectional_dynamic_rnn(input_tensors, forward_kernel, forward_bias, backward_kernel, backward_bias):
    """
    inference logic of a bidirectional lstm

    :param input_tensors: a list of input tensor
    :param forward_kernel: kernel weight of forward cell
    :param forward_bias:  kernel bias of forward cell
    :param backward_kernel: kernel weight of backward cell
    :param backward_bias: kernel bias of backward cell
    :return: forward_hidden, backward_hidden
    """

    # reverse input tensors
    inv_input_tensors = input_tensors[::-1]

    # forward and backward
    forward_hidden_lst, _ = dynamic_rnn(input_tensors, forward_kernel, forward_bias)
    backward_hidden_lst, _ = dynamic_rnn(inv_input_tensors, backward_kernel, backward_bias)

    # reverse backward hidden
    backward_hidden_lst.reverse()

    return np.array(forward_hidden_lst), np.array(backward_hidden_lst)

def stack_bidirectional_dynamic_rnn(inps, forward_kernel_lst, forward_bias_lst, backward_kernel_lst, backward_bias_lst):
    """
    inference logic of a stack bidirectional lstm

    :param input_tensors: a list of input tensor
    :param forward_kernel_lst: kernel weight of forward cell for each layer
    :param forward_bias_lst:  kernel bias of forward cell for each layer
    :param backward_kernel_lst: kernel weight of backward cell for each layer
    :param backward_bias_lst: kernel bias of backward cell for each layer
    :return: combined hiddens
    """

    layer_size = len(forward_kernel_lst)

    # check the number of layer is same
    assert len(forward_bias_lst) == layer_size
    assert len(backward_kernel_lst) == layer_size
    assert len(backward_bias_lst) == layer_size

    # the shape of first layer is different from other layers, run it separately
    forward_hidden, backward_hidden = bidirectional_dynamic_rnn(inps, forward_kernel_lst[0], forward_bias_lst[0], backward_kernel_lst[0], backward_bias_lst[0])

    for i in range(1, layer_size):
        hiddens = np.concatenate([forward_hidden, backward_hidden], axis=1)

        forward_hidden, backward_hidden = bidirectional_dynamic_rnn(hiddens, forward_kernel_lst[i], forward_bias_lst[i], backward_kernel_lst[i], backward_bias_lst[i])

    return np.concatenate([forward_hidden, backward_hidden], axis=1)