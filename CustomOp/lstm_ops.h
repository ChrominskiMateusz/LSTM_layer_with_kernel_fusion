/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_RNN_KERNELS_LSTM_OPS_H_
#define TENSORFLOW_CONTRIB_RNN_KERNELS_LSTM_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

#include "TF_headers/eigen_activations.h"
#include "TF_headers/blas_gemm.h"

static const int BATCH_SIZE = 128;

template<typename T>
void make_sparse (Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> matrix, 
                  const int group_size, 
                  const int start, 
                  const int end,
                  Eigen::TensorMap<Eigen::Tensor<long long, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> indices,
                  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> values,
                  const int part)
{
  int max;
  int counter{part * BATCH_SIZE * BATCH_SIZE / 2 / group_size};
  
  int offset{part % 2 ? part - 1 : part};
  offset *= BATCH_SIZE / 2;
  
  int end_ = start ? BATCH_SIZE : BATCH_SIZE / 2;

  for (int i{}; i < BATCH_SIZE; i++)
    for (int j{start ? BATCH_SIZE / 2 : start}; j + group_size <= end_; j += group_size)
    {
      max = 0;
    
      for (int k{1}; k < group_size; k++)
        if (fabs (float (matrix(i, j + max))) < fabs (float (matrix(i, j + k))))
          max = k;

      for (int k{}; k < group_size; k++)
        if (k != max)
          matrix(i, j + k) = (T)NULL;

      values(counter) = matrix(i, j + max);
      indices(counter, 0) = i;
      indices(counter, 1) = j + max + offset;

      counter++;
    }
}

namespace tensorflow {
class OpKernelContext;

namespace functor {

template <typename Device, typename T>
struct TensorZero {
  void operator()(const Device& d, typename TTypes<T>::Flat t) {
    t.device(d) = t.constant(T(0));
  }
};

template <typename Device, typename T>
struct TensorUnalignedZero {
  void operator()(const Device& d, typename TTypes<T>::UnalignedFlat t) {
    t.device(d) = t.constant(T(0));
  }
};

template <typename Device, typename T>
struct TensorCopy {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat src,
                  typename TTypes<T>::Flat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorCopyUnaligned {
  void operator()(const Device& d, typename TTypes<T>::UnalignedConstFlat src,
                  typename TTypes<T>::Flat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorCopyToUnaligned {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat src,
                  typename TTypes<T>::UnalignedFlat dst) {
    dst.device(d) = src;
  }
};

template <typename Device, typename T>
struct TensorAdd {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat a,
                  typename TTypes<T>::ConstFlat b, typename TTypes<T>::Flat c) {
    c.device(d) = a + b;
  }
};

template <typename Device, typename T>
struct TensorZeroPadding {
  void operator()(const Device& d, const int64 time_idx,
                  typename TTypes<int64>::ConstVec seq_len,
                  typename TTypes<T>::Vec mask, typename TTypes<T>::Matrix m) {
    // mask is shape [batch_size].
    mask.device(d) = seq_len.constant(time_idx) < seq_len;

    // m_shape is [batch_size, 1].
    Eigen::array<Eigen::DenseIndex, 2> m_shape({m.dimensions()[0], 1});
    // broadcast_shape is [1, units].
    Eigen::array<Eigen::DenseIndex, 2> broadcast_shape({1, m.dimensions()[1]});

    // m is shape [batch_size, units].
    m.device(d) = m * mask.reshape(m_shape).broadcast(broadcast_shape);
  }
};

struct LSTMBlockCell {
  LSTMBlockCell(const int batch_size, const int input_size, const int cell_size)
      : batch_size_(batch_size),
        input_size_(input_size),
        cell_size_(cell_size) {}

  int batch_size() const { return batch_size_; }

  int input_size() const { return input_size_; }

  int cell_size() const { return cell_size_; }

  inline Eigen::array<Eigen::DenseIndex, 2> icfo_i_offsets() const {
    return {0, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> icfo_c_offsets() const {
    return {0, cell_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> icfo_f_offsets() const {
    return {0, cell_size_ * 2};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> icfo_o_offsets() const {
    return {0, cell_size_ * 3};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> cell_extents() const {
    return {batch_size_, cell_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_x_offsets() const {
    return {0, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_x_extents() const {
    return {batch_size_, input_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_h_offsets() const {
    return {0, input_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> xh_h_extents() const {
    return {batch_size_, cell_size_};
  }

 protected:
  const int batch_size_;
  const int input_size_;
  const int cell_size_;
};

// See lstm_ops.cc for CPUDevice implementation and lstm_ops_gpu.cu.cc for
// GPUDevice implementation.
template <typename Device, typename T, bool USE_CUBLAS>
struct LSTMBlockCellFprop : public LSTMBlockCell {
  LSTMBlockCellFprop(const int batch_size, const int input_size,
                     const int cell_size)
      : LSTMBlockCell(batch_size, input_size, cell_size) {}

  void operator()(OpKernelContext* ctx, const Device& d,
                  const float forget_bias, const float cell_clip,
                  bool use_peephole, typename TTypes<T>::ConstMatrix x,
                  typename TTypes<T>::ConstMatrix cs_prev,
                  typename TTypes<T>::ConstMatrix h_prev,
                  typename TTypes<T>::ConstMatrix w,
                  typename TTypes<T>::ConstVec wci,
                  typename TTypes<T>::ConstVec wcf,
                  typename TTypes<T>::ConstVec wco,
                  typename TTypes<T>::ConstVec b, typename TTypes<T>::Matrix xh,
                  typename TTypes<T>::Matrix i, typename TTypes<T>::Matrix cs,
                  typename TTypes<T>::Matrix f, typename TTypes<T>::Matrix o,
                  typename TTypes<T>::Matrix ci, typename TTypes<T>::Matrix co,
                  typename TTypes<T>::Matrix icfo,
                  typename TTypes<T>::Matrix h);
};

template <typename Device, typename T, bool USE_CUBLAS>
struct BlockLSTMBprop : public LSTMBlockCell {
  BlockLSTMBprop(const int batch_size, const int input_size,
                 const int cell_size)
      : LSTMBlockCell(batch_size, input_size, cell_size) {}

  void operator()(
      OpKernelContext* ctx, const Device& d, bool use_peephole,
      typename TTypes<T>::ConstMatrix x,
      typename TTypes<T>::ConstMatrix cs_prev,
      typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
      typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
      typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
      typename TTypes<T>::Matrix xh, typename TTypes<T>::ConstMatrix i,
      typename TTypes<T>::ConstMatrix cs, typename TTypes<T>::ConstMatrix f,
      typename TTypes<T>::ConstMatrix o, typename TTypes<T>::ConstMatrix ci,
      typename TTypes<T>::ConstMatrix co,
      typename TTypes<T>::ConstMatrix cs_grad,
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,
      typename TTypes<T>::Matrix dicfo, typename TTypes<T>::Matrix cs_prev_grad,
      typename TTypes<T>::Matrix h_prev_grad,
      typename TTypes<T>::Matrix xh_grad, typename TTypes<T>::Matrix x_grad,
      typename TTypes<T>::Matrix w_grad, typename TTypes<T>::Vec wci_grad,
      typename TTypes<T>::Vec wcf_grad, typename TTypes<T>::Vec wco_grad,
      typename TTypes<T>::Vec b_grad,
      typename TTypes<T>::Vec values,
      typename TTypes<int64>::Matrix indices) {
    // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
    do_.device(d) = o * (o.constant(T(1)) - o) * h_grad * co;

    // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t] + dcs[t + 1] .* f[t + 1]
    dcs.device(d) = (co.constant(T(1)) - co * co) * h_grad * o + cs_grad;

    Eigen::array<Eigen::DenseIndex, 2> p_shape({1, cell_size_});
    Eigen::array<Eigen::DenseIndex, 2> p_broadcast_shape({batch_size_, 1});
    if (use_peephole) {
      dcs.device(d) =
          dcs + do_ * wco.reshape(p_shape).broadcast(p_broadcast_shape);
    }

    // dci[t] = tanh'(ci[t]) dcs[t] i[t]
    dci.device(d) = (ci.constant(T(1)) - ci * ci) * dcs * i;

    // df[t] = sigm'(f[t]) dcs[t] cs[t - 1]
    df.device(d) = f * (f.constant(T(1)) - f) * dcs * cs_prev;

    // di[t] = sigm'(i[t]) dcs[t] ci[t]
    di.device(d) = i * (i.constant(T(1)) - i) * dcs * ci;

    const int START = 0;
    const int GROUP_SIZE = 16;

    std::thread di_thread (&make_sparse<T>, di, GROUP_SIZE, START, di.size () / 2, indices, values, 0);
    std::thread di_thread2 (&make_sparse<T>, di, GROUP_SIZE, di.size () / 2, di.size (), indices, values, 1);

    std::thread dci_thread (&make_sparse<T>, dci, GROUP_SIZE, START, dci.size () / 2, indices, values, 2);
    std::thread dci_thread2 (&make_sparse<T>, dci, GROUP_SIZE, dci.size () / 2, dci.size (), indices, values, 3);

    std::thread df_thread (&make_sparse<T>, df, GROUP_SIZE, START, df.size () / 2, indices, values, 4);
    std::thread df_thread2 (&make_sparse<T>, df, GROUP_SIZE, df.size () / 2, df.size (), indices, values, 5);
    
    std::thread do__thread (&make_sparse<T>, do_, GROUP_SIZE, START, do_.size () / 2, indices, values, 6);
    std::thread do__thread2 (&make_sparse<T>, do_, GROUP_SIZE, do_.size () / 2, do_.size (), indices, values, 7);

    di_thread.join ();
    di_thread2.join ();
    dicfo.slice(icfo_i_offsets(), cell_extents()).device(d) = di;

    dci_thread.join ();
    dci_thread2.join ();
    dicfo.slice(icfo_c_offsets(), cell_extents()).device(d) = dci;

    df_thread.join ();
    df_thread2.join ();
    dicfo.slice(icfo_f_offsets(), cell_extents()).device(d) = df;

    do__thread.join ();
    do__thread2.join ();
    dicfo.slice(icfo_o_offsets(), cell_extents()).device(d) = do_;

    cs_prev_grad.device(d) = dcs * f;
    if (use_peephole) {
      cs_prev_grad.device(d) =
          cs_prev_grad +
          di * wci.reshape(p_shape).broadcast(p_broadcast_shape) +
          df * wcf.reshape(p_shape).broadcast(p_broadcast_shape);
    }

    // xh_grad.
    typename TTypes<T>::ConstMatrix const_dicfo(dicfo.data(),
                                                dicfo.dimensions());
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, false, true, 1.f, const_dicfo, w, 0.f, xh_grad);

    // xh.
    xh.slice(xh_x_offsets(), xh_x_extents()).device(d) = x;
    xh.slice(xh_h_offsets(), xh_h_extents()).device(d) = h_prev;
    typename TTypes<T>::ConstMatrix const_xh(xh.data(), xh.dimensions());

    // x_grad.
    x_grad.device(d) = xh_grad.slice(xh_x_offsets(), xh_x_extents());
    h_prev_grad.device(d) = xh_grad.slice(xh_h_offsets(), xh_h_extents());

    // w_grad.
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, true, false, 1.f, const_xh, const_dicfo, 1.f, w_grad);

    // b_grad.
    b_grad.device(d) += dicfo.sum(Eigen::array<int, 1>({0}));

    if (use_peephole) {
      wci_grad.device(d) += (di * cs_prev).sum(Eigen::array<int, 1>({0}));
      wcf_grad.device(d) += (df * cs_prev).sum(Eigen::array<int, 1>({0}));
      wco_grad.device(d) += (do_ * cs).sum(Eigen::array<int, 1>({0}));
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_RNN_KERNELS_LSTM_OPS_H_
