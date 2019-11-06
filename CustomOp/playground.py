import tensorflow as tf
import time as time

sess = tf.InteractiveSession()

num_units = 128 * 100
batch_size = 128 * 100
input_size = 28

# Init with floats from 0 to 1
x = tf.random_uniform([batch_size, num_units * 4], minval=0, maxval=1, dtype=tf.float32)                # icfo
y = tf.random_uniform([batch_size + input_size, num_units * 4], minval=0, maxval=1, dtype=tf.float32)   # w
y_t = tf.transpose(y)                                                                                   # transpose w

# k - number of largest elements that will remain in each row
k = num_units * 4 // 16 
values, indices = tf.nn.top_k(x, k, sorted=False)
values = tf.reshape(values, [-1])

# print(x.eval())
# print(values.eval())
# print(indices.eval())

my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)
my_range_repeated = tf.tile(my_range, [1, k])

full_indices = tf.concat([tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)], axis=2)
full_indices = tf.reshape(full_indices, [-1, 2])
#print(full_indices.eval())

# Allocate SparseTensor
sparse_x = tf.SparseTensor(indices=tf.cast(full_indices, dtype=tf.int64), values=values, dense_shape=x.shape)
#print(sparse_x.eval())

# Usage of standard Tensorflow matrix multiplication
start_time = time.time()
resStandardMatMul = tf.matmul(x, y, transpose_b=True)
duration = time.time() - start_time
print('Standard matmul: {}'.format(duration))

# Sparse matrix multiplication
start_time = time.time()
resSparseMatMul0 = tf.sparse.sparse_dense_matmul(sparse_x, tf.transpose(y))
duration = time.time() - start_time
print('Sparse matmul tf.transpose: {}'.format(duration))

start_time = time.time()
resSparseMatMul1 = tf.sparse.sparse_dense_matmul(sparse_x, y, adjoint_b=True)
duration = time.time() - start_time
print('Sparse matmul adjoint_b=True, flag: {}'.format(duration))

start_time = time.time()
resSparseMatMul2 = tf.sparse.sparse_dense_matmul(sparse_x, y_t)
duration = time.time() - start_time
print('Sparse matmul without transposing: {}'.format(duration))

# Tensor dimensions:
#
# w[num_units + input_size][num_units * 4]
# icfo[batch_size][num_units * 4]
#
# U nas:
# num_units = 128
# batch_size = 128
# input_size = 28