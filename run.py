#!/usr/bin/env python
import sys

import numpy
import theano
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent
from blocks.utils import pack

dim = eval(sys.argv[2])
transition = eval(sys.argv[1])(dim=dim, activation=Tanh())
batch_size = 10
n_steps = 50

input_vars = {name: tensor.tensor3(name)
              for name in transition.apply.sequences
              if name != 'mask'}
output_vars = pack(transition.apply(**input_vars))
cost = sum(output.sum() for output in output_vars)
grads = tensor.grad(cost, list(transition.params))
function = theano.function(input_vars.values(), grads)
inputs = {name: numpy.random.rand(n_steps, batch_size, transition.get_dim(name))
          for name in input_vars}
function(**inputs)


