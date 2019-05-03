# implement tensorflow multilayer blstm with numpy
* This repo implements numpy inference logic of tensorflow stack_bidirectional_dynamic_rnn with cudnn cell
* The original purpose of this project is to export my cudnn acoustic model to numpy, so only cudnn related cell features would be supported. Some features in LSTMBlockCell are not supported (e.g: pee hole, clip cell) 
* check model.py for numpy implementation
* check test.py for equivalent tensorflow cudnn implementation
* Some details of this implementation is described [here](https://www.xinjianl.com/blog/2019/05/03/export-cudnn-blstm-to-numpy/) 