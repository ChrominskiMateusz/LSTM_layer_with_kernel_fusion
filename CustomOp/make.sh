TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++-4.8 -std=c++11 -shared lstm_ops.cc -o lstm_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -Ofast -pthread