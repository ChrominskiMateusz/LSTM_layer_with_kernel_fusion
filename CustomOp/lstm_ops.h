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
#include "tensorflow/core/framework/bounds_check.h"

#include "TF_headers/eigen_activations.h"
#include "TF_headers/blas_gemm.h"

template<typename T>
void add_to_sparse (Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> matrix, 
                  int& counter, 
                  const int x_, 
                  const int y_, 
                  const int offset,
                  Eigen::TensorMap<Eigen::Tensor<long long, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> indices,
                  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> values)
{
  values(counter) = matrix(x_, y_);
  indices(counter, 0) = x_;
  indices(counter, 1) = y_ + offset;
  counter++;
}

template<typename T>
void make_sparse (Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> matrix, 
                  const int group_size, 
                  const int start, 
                  const int end,
                  Eigen::TensorMap<Eigen::Tensor<long long, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> indices,
                  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> values,
                  const int part,
                  const int thread_count)
{
  /*
  matrix - matrix to sparse
  group_size - size of every group from which we will be choosin' max value
  start - start of matrix.dimension(1) 
  end - end of matrix.dimension(1)
  indices - indices of max elements
  values - values of max elements
  part - index of function used to fill indices/values
  thread_count - how many threads in use for this matrix
  */

  const int x_ = matrix.dimension(0);
  const int y_ = matrix.dimension(1);

  int split_offset = start ? (group_size - (x_ * start) % group_size) % group_size : start;

  int counter{part * x_ * y_ / thread_count / group_size};
  const int end_ = start + y_ / thread_count;
  
  int offset{part - part % thread_count};
  offset *= y_ / thread_count;
  
  int max;
  int j;

  for (int i{}; i < x_; i++)
  {
    for (j = start + split_offset; j + group_size <= end_; j += group_size)
    {
      max = 0;
    
      for (int k{1}; k < group_size; k++)
        if (fabs (float (matrix(i, j + max))) < fabs (float (matrix(i, j + k))))
          max = k;

      // for (int k{}; k < group_size; k++)
      //   if (k != max)
      //     matrix(i, j + k) = (T)NULL;

      add_to_sparse(matrix, counter, i, j + max, offset, indices, values);
    }
    
    split_offset = (j == end_) ? 0 : j + group_size - end_;
    if (split_offset)
      add_to_sparse(matrix, counter, i, end_ - 1, offset, indices, values);
  }
}


template<typename T>
void one_thread_sparse (Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> matrix,
                  Eigen::TensorMap<Eigen::Tensor<long long, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> indices,
                  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> values, 
                  const int group_size)
{
  const int x_ = matrix.dimension(0);
  const int y_ = matrix.dimension(1);

  int counter{};
  int split_offset{};
  int j;
  int max;

  for (int i{}; i < x_; i++)
  {
    for (j = split_offset; j + group_size <= y_; j += group_size)
    {
      max = 0;
    
      for (int k{1}; k < group_size; k++)
        if (fabs (float (matrix(i, j + max))) < fabs (float (matrix(i, j + k))))
          max = k;

      add_to_sparse(matrix, counter, i, j + max, 0, indices, values);
    }
    
    split_offset = (j == y_) ? 0 : j + group_size - y_;
    if (split_offset)
      add_to_sparse(matrix, counter, i, y_ - 1, 0, indices, values);
  }
}


namespace tensorflow {

class OpKernelContext;

namespace functor {

template <typename MATRIX, bool ADJ>
class MaybeAdjoint;

template <typename MATRIX>
class MaybeAdjoint<MATRIX, false> {
 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MaybeAdjoint(MATRIX m) : m_(m) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename MATRIX::Scalar operator()(
      const typename MATRIX::Index i, const typename MATRIX::Index j) const {
    return m_(i, j);
  }

 private:
  const MATRIX m_;
};

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T MaybeConj(T v) {
  return v;
}

template <typename MATRIX>
class MaybeAdjoint<MATRIX, true> {
 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MaybeAdjoint(MATRIX m) : m_(m) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename MATRIX::Scalar operator()(
      const typename MATRIX::Index i, const typename MATRIX::Index j) const {
    return Eigen::numext::conj(m_(j, i));
  }

 private:
  const MATRIX m_;
};

template<typename Device, typename T, bool ADJ_A, bool ADJ_B> 
void sparse_dense_matmul(const Device& d,
                         typename TTypes<T>::Matrix out,
                         typename TTypes<int64>::ConstMatrix a_indices,
                         typename TTypes<T>::ConstVec a_values,
                         typename TTypes<T>::ConstMatrix b)
{
  const std::size_t kNumVectorize = 32;
  const std::size_t nnz = a_values.size();
  const std::size_t rhs_right = (ADJ_B ? b.dimension(0) : b.dimension(1));
  const std::size_t lhs_right = (ADJ_B ? b.dimension(1) : b.dimension(0));
  const int lhs_index_a = ADJ_A ? 1 : 0;
  const int rhs_index_a = ADJ_A ? 0 : 1;

  out.setZero();

  if (rhs_right < kNumVectorize)
  {
    auto maybe_adjoint_b = MaybeAdjoint<decltype(b), ADJ_B>(b);
    for(std::size_t i = 0; i < nnz; ++i)
    {
      const int64 m = internal::SubtleMustCopy(a_indices(i, lhs_index_a));
      const int64 k = internal::SubtleMustCopy(a_indices(i, rhs_index_a));
      if (!FastBoundsCheck(k, lhs_right)) 
      {
        throw "Sparse Matmul dimensions not match";
        return;
      }
      if (!FastBoundsCheck(m, out.dimension(0))) 
      {
        throw "Sparse Matmul dimensions not match";
        return;
      }

      const T a_value = ADJ_A ? MaybeConj(a_values(i)) : a_values(i);
      for (std::size_t n = 0; n < rhs_right; ++n)
      {
        const T b_value = maybe_adjoint_b(k, n);
        out(m, n) += a_value * b_value;
      }
    }
  }
  else
  {
    // Vectorization via Eigen.
    const int b_chip_index = ADJ_B ? 1 : 0;
    #define LOOP_NNZ(b_passed)                                                         \
    for (std::size_t i = 0; i < nnz; ++i) {                                            \
      const int64 m = internal::SubtleMustCopy(a_indices(i, lhs_index_a));             \
      const int64 k = internal::SubtleMustCopy(a_indices(i, rhs_index_a));             \
      const T a_value = (ADJ_A) ? MaybeConj(a_values(i)) : a_values(i);                \
      if (!FastBoundsCheck(k, lhs_right)) {                                            \
        throw "Sparse Matmul dimensions not match";                                    \
        return;                                                                        \
      }                                                                                \
      if (!FastBoundsCheck(m, out.dimension(0))) {                                     \
        throw "Sparse Matmul dimensions not match";                                    \
        return;                                                                        \
      }                                                                                \
      out.template chip<0>(m) +=  b_passed.template chip<b_chip_index>(k) * a_value;   \
    }

    if(ADJ_B)
    {
      // Perform transpose and conjugation on B once, since we chip out B's
      // columns in the nnz loop.
      Eigen::array<int, 2> shuffle{ {1, 0} };  // preserve dimension order

      Eigen::Tensor<T, 2, Eigen::ColMajor> col_major_conj_b =
        b.swap_layout().shuffle(shuffle).conjugate();
        
      LOOP_NNZ(col_major_conj_b);
    }
    else
    {
      LOOP_NNZ(b);
    }
    #undef LOOP_NNZ
  }

}

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
                 const int cell_size, const int group_size)
      : LSTMBlockCell(batch_size, input_size, cell_size),
      group_size (group_size) {}

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
      typename TTypes<int64>::Matrix indices,
      typename TTypes<T>::Vec svalues,
      typename TTypes<int64>::Matrix sindices) {
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

    dicfo.slice(icfo_i_offsets(), cell_extents()).device(d) = di;;
    dicfo.slice(icfo_c_offsets(), cell_extents()).device(d) = dci;
    dicfo.slice(icfo_f_offsets(), cell_extents()).device(d) = df;
    dicfo.slice(icfo_o_offsets(), cell_extents()).device(d) = do_;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    one_thread_sparse(dicfo, indices, values, group_size);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "First make sparse = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

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

    typename TTypes<int64>::ConstMatrix const_indices(indices.data(), indices.dimensions());
    typename TTypes<T>::ConstVec const_values(values.data(), values.dimensions());

    begin = std::chrono::steady_clock::now();
    // Dense matmul                               
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
       ctx, d, false, true, 1.f, const_dicfo, w, 0.f, xh_grad);
    end = std::chrono::steady_clock::now();
    std::cout << "First dense matmul = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();
    // Sparse dense matmul
    sparse_dense_matmul<Device, T, false, true>(d, xh_grad, const_indices, const_values, w);
    end = std::chrono::steady_clock::now();
    std::cout << "First sparse matmul = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    

    // xh.
    xh.slice(xh_x_offsets(), xh_x_extents()).device(d) = x;
    xh.slice(xh_h_offsets(), xh_h_extents()).device(d) = h_prev;

    begin = std::chrono::steady_clock::now();
    one_thread_sparse(xh, sindices, svalues, group_size);
    end = std::chrono::steady_clock::now();
    std::cout << "Second make sparse = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    typename TTypes<T>::ConstMatrix const_xh(xh.data(), xh.dimensions());

    // x_grad.
    x_grad.device(d) = xh_grad.slice(xh_x_offsets(), xh_x_extents());
    h_prev_grad.device(d) = xh_grad.slice(xh_h_offsets(), xh_h_extents());
    
    typename TTypes<int64>::ConstMatrix const_sindices(sindices.data(), sindices.dimensions());
    typename TTypes<T>::ConstVec const_svalues(svalues.data(), svalues.dimensions());

    begin = std::chrono::steady_clock::now();
    // w_grad.
    TensorBlasGemm<Device, T, USE_CUBLAS>::compute(
        ctx, d, true, false, 1.f, const_xh, const_dicfo, 1.f, w_grad);
    end = std::chrono::steady_clock::now();
    std::cout << "Second dense matmul = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();
    // Sparse dense matmul
    sparse_dense_matmul<Device, T, true, false>(d, w_grad, const_sindices, const_svalues, const_dicfo);
    end = std::chrono::steady_clock::now();
    std::cout << "Second sparse matmul = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]\n\n";

    // b_grad.
    b_grad.device(d) += dicfo.sum(Eigen::array<int, 1>({0}));

    if (use_peephole) {
      wci_grad.device(d) += (di * cs_prev).sum(Eigen::array<int, 1>({0}));
      wcf_grad.device(d) += (df * cs_prev).sum(Eigen::array<int, 1>({0}));
      wco_grad.device(d) += (do_ * cs).sum(Eigen::array<int, 1>({0}));
    }
  }
  private:
  const int group_size;
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_RNN_KERNELS_LSTM_OPS_H_
