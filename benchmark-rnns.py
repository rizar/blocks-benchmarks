#!/usr/bin/env python
import sys
import timeit

import numpy
import theano
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent
from blocks.utils import pack

batch_size = 10
n_steps = 50
transitions = [SimpleRecurrent, GatedRecurrent, LSTM]
dims = [100, 250, 1000, 2000]



table = []
for transition in transitions:
    row = []
    for dim in dims:
        brick = transition(dim=dim, activation=Tanh())
        input_vars = {name: tensor.tensor3(name)
                    for name in brick.apply.sequences
                    if name != 'mask'}
        output_vars = pack(brick.apply(**input_vars))
        cost = sum(output.sum() for output in output_vars)
        grads = tensor.grad(cost, list(brick.params))
        function = theano.function(input_vars.values(), grads)
        inputs = {name: numpy.random.rand(n_steps, batch_size, brick.get_dim(name))
                .astype(theano.config.floatX)
                for name in input_vars}

        result = timeit.timeit(lambda: function(**inputs), number=5) / 5
        print transition.__name__, dim, result
        row.append(result)
    table.append(row)

sep = '|'
print sep, sep, sep.join(str(dim) for dim in dims), sep
print '---'.join(sep * (len(dims) + 2))
for transition, row in zip(transitions, table):
    print sep, sep.join([transition.__name__] + [str(int(r * 1000 + 0.5)) for r in row]), sep

