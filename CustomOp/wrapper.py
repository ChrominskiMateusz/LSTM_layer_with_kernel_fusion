# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""LSTM Block Cell ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl

_lstm_ops_so = tf.load_op_library('./lstm_ops.so')

LayerRNNCell = rnn_cell_impl.LayerRNNCell  # pylint: disable=invalid-name


# pylint: disable=invalid-name
def _lstm_block_cell(x,
                     cs_prev,
                     h_prev,
                     w,
                     b,
                     wci=None,
                     wcf=None,
                     wco=None,
                     forget_bias=None,
                     cell_clip=None,
                     use_peephole=None,
                     sparse_bprop=None,
                     name=None):
  r"""Computes the LSTM cell forward propagation for 1 time step.

  This implementation uses 1 weight matrix and 1 bias vector, and there's an
  optional peephole connection.

  This kernel op implements the following mathematical equations:

  ```python
  xh = [x, h_prev]
  [i, ci, f, o] = xh * w + b
  f = f + forget_bias

  if not use_peephole:
    wci = wcf = wco = 0

  i = sigmoid(cs_prev * wci + i)
  f = sigmoid(cs_prev * wcf + f)
  ci = tanh(ci)

  cs = ci .* i + cs_prev .* f
  cs = clip(cs, cell_clip)

  o = sigmoid(cs * wco + o)
  co = tanh(cs)
  h = co .* o
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
      The input to the LSTM cell, shape (batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the cell state at previous time step.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Output of the previous cell at previous time step.
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    forget_bias: An optional `float`. Defaults to `1`. The forget gate bias.
    cell_clip: An optional `float`. Defaults to `-1` (no clipping).
      Value to clip the 'cs' value to. Disable by setting to negative value.
    use_peephole: An optional `bool`. Defaults to `False`.
      Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).
    i: A `Tensor`. Has the same type as `x`. The input gate.
    cs: A `Tensor`. Has the same type as `x`. The cell state before the tanh.
    f: A `Tensor`. Has the same type as `x`. The forget gate.
    o: A `Tensor`. Has the same type as `x`. The output gate.
    ci: A `Tensor`. Has the same type as `x`. The cell input.
    co: A `Tensor`. Has the same type as `x`. The cell after the tanh.
    h: A `Tensor`. Has the same type as `x`. The output h vector.

  Raises:
    ValueError: If cell_size is None.
  """
  if wci is None:
    cell_size = cs_prev.get_shape().with_rank(2).dims[1].value
    if cell_size is None:
      raise ValueError("cell_size from `cs_prev` should not be None.")
    wci = array_ops.constant(0, dtype=dtypes.float32, shape=[cell_size])
    wcf = wci
    wco = wci

  # pylint: disable=protected-access
  return _lstm_ops_so.lstm_block_cell_our(
      x=x,
      cs_prev=cs_prev,
      h_prev=h_prev,
      w=w,
      wci=wci,
      wcf=wcf,
      wco=wco,
      b=b,
      forget_bias=forget_bias,
      cell_clip=cell_clip if cell_clip is not None else -1,
      use_peephole=use_peephole,
      sparse_bprop=sparse_bprop,
      name=name)
  # pylint: enable=protected-access


@ops.RegisterGradient("LSTMBlockCellOur")
def _LSTMBlockCellGrad(op, *grad):
  """Gradient for LSTMBlockCell."""
  (x, cs_prev, h_prev, w, wci, wcf, wco, b) = op.inputs
  (i, cs, f, o, ci, co, _) = op.outputs
  (_, cs_grad, _, _, _, _, h_grad) = grad

  batch_size = x.get_shape().with_rank(2).dims[0].value
  if batch_size is None:
    batch_size = -1
  input_size = x.get_shape().with_rank(2).dims[1].value
  if input_size is None:
    raise ValueError("input_size from `x` should not be None.")
  cell_size = cs_prev.get_shape().with_rank(2).dims[1].value
  if cell_size is None:
    raise ValueError("cell_size from `cs_prev` should not be None.")

  (cs_prev_grad, dicfo, wci_grad, wcf_grad,
   wco_grad) = _lstm_ops_so.lstm_block_cell_grad_our(
       x,
       cs_prev,
       h_prev,
       w,
       wci,
       wcf,
       wco,
       b,
       i,
       cs,
       f,
       o,
       ci,
       co,
       cs_grad,
       h_grad,
       use_peephole=op.get_attr("use_peephole"),
       sparse_bprop=op.get_attr("sparse_bprop"))

  sparse_bprop=op.get_attr("sparse_bprop")

  if sparse_bprop:
    # dicfo sparsification
    k = 102
    values, indices = tf.nn.top_k(dicfo, k, sorted=False)

    # Make values flat
    values = tf.reshape(values, [-1])

    my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)
    my_range_repeated = tf.tile(my_range, [1, k])

    full_indices = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)], axis=2)
    full_indices = tf.reshape(full_indices, [-1, 2])

    sparse_dicfo = tf.SparseTensor(
            indices=tf.cast(full_indices, dtype=tf.int64),
            values=values,
            dense_shape=dicfo.shape)

    xh_grad = tf.sparse.sparse_dense_matmul(sparse_dicfo, w, adjoint_b=True)


  # Backprop from dicfo to xh.
  #dicfo = tf.Print(dicfo, [dicfo], "DICFO: ", summarize=2)

  if sparse_bprop == False:
    xh_grad = math_ops.matmul(dicfo, w, transpose_b=True)

  x_grad = array_ops.slice(xh_grad, (0, 0), (batch_size, input_size))
  x_grad.get_shape().merge_with(x.get_shape())

  h_prev_grad = array_ops.slice(xh_grad, (0, input_size),(batch_size, cell_size))
  h_prev_grad.get_shape().merge_with(h_prev.get_shape())

  # Backprop from dicfo to w.
  xh = array_ops.concat([x, h_prev], 1)

  w_grad = math_ops.matmul(xh, dicfo, transpose_a=True)
  w_grad.get_shape().merge_with(w.get_shape())

  # Backprop from dicfo to b.
  b_grad = nn_ops.bias_add_grad(dicfo)
  b_grad.get_shape().merge_with(b.get_shape())

  return (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad,
          wco_grad, b_grad)


class LSTMBlockCell(LayerRNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add `forget_bias` (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  Unlike `rnn_cell_impl.LSTMCell`, this is a monolithic op and should be much
  faster.  The weight and bias matrices should be compatible as long as the
  variable scope matches.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               cell_clip=None,
               use_peephole=False,
               sparse_bprop=False,
               dtype=None,
               reuse=None,
               name="lstm_cell"):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      cell_clip: An optional `float`. Defaults to `-1` (no clipping).
      use_peephole: Whether to use peephole connections or not.
      dtype: the variable dtype of this layer. Default to tf.float32.
      reuse: (optional) boolean describing whether to reuse variables in an
        existing scope.  If not `True`, and the existing scope already has the
        given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.  By default this is "lstm_cell", for variable-name compatibility
        with `tf.compat.v1.nn.rnn_cell.LSTMCell`.

      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMBlockCell instead.
    """
    super(LSTMBlockCell, self).__init__(_reuse=reuse, dtype=dtype, name=name)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._use_peephole = use_peephole
    self._sparse_bprop = sparse_bprop
    self._cell_clip = cell_clip if cell_clip is not None else -1
    self._names = {
        "W": "kernel",
        "b": "bias",
        "wci": "w_i_diag",
        "wcf": "w_f_diag",
        "wco": "w_o_diag",
        "scope": "lstm_cell"
    }
    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if not inputs_shape.dims[1].value:
      raise ValueError(
          "Expecting inputs_shape[1] to be set: %s" % str(inputs_shape))
    input_size = inputs_shape.dims[1].value
    self._kernel = self.add_variable(
        self._names["W"], [input_size + self._num_units, self._num_units * 4])
    self._bias = self.add_variable(
        self._names["b"], [self._num_units * 4],
        initializer=init_ops.constant_initializer(0.0))
    if self._use_peephole:
      self._w_i_diag = self.add_variable(self._names["wci"], [self._num_units])
      self._w_f_diag = self.add_variable(self._names["wcf"], [self._num_units])
      self._w_o_diag = self.add_variable(self._names["wco"], [self._num_units])

    self.built = True

  def call(self, inputs, state):
    """Long short-term memory cell (LSTM)."""
    if len(state) != 2:
      raise ValueError("Expecting state to be a tuple with length 2.")

    if self._use_peephole:
      wci = self._w_i_diag
      wcf = self._w_f_diag
      wco = self._w_o_diag
    else:
      wci = wcf = wco = array_ops.zeros([self._num_units], dtype=self.dtype)

    (cs_prev, h_prev) = state
    (_, cs, _, _, _, _, h) = _lstm_block_cell(
        inputs,
        cs_prev,
        h_prev,
        self._kernel,
        self._bias,
        wci=wci,
        wcf=wcf,
        wco=wco,
        forget_bias=self._forget_bias,
        cell_clip=self._cell_clip,
        use_peephole=self._use_peephole,
        sparse_bprop=self._sparse_bprop)

    new_state = rnn_cell_impl.LSTMStateTuple(cs, h)
    return h, new_state