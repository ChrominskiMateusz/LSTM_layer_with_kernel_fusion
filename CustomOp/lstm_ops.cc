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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "lstm_ops.h"

#include <memory>
#include <vector>
#include <iostream>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include <fstream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

template<typename T>
const T count_delete_point (Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> matrix, const float& percentage, const T& average)
{
	std::vector<T> elements;
	for (int i{}; i < matrix.size (); i++)
		elements.push_back (matrix.data ()[i]);

	std::sort (elements.begin (), elements.end (), [&average](T a, T b) {
		return abs (a - average) < abs (b - average);
	});

	return abs (elements[int (elements.size () * percentage)] - average);
}

template<typename T>
const T count_average (Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> matrix)
{
	T sum{};  // nie wiem czemu tu kurwa nie działa normalne matrix.sum () xD coś znowu z typami 
	for (int i{}; i < matrix.size (); i++)
		sum += matrix.data ()[i];

	return sum / matrix.size ();
}

template<typename T>
void make_sparse (Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> matrix, const float& percentage)
{
	const T average = count_average (matrix);
	const T delete_point = count_delete_point (matrix, percentage, average);
	
	for (int i{}; i < matrix.size (); i++)
		if (abs (matrix.data ()[i] - average) > delete_point)
			matrix.data ()[i] = (T)NULL;
}

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("LSTMBlockCellOur")
    .Input("x: T")
    .Input("cs_prev: T")
    .Input("h_prev: T")
    .Input("w: T")
    .Input("wci: T")
    .Input("wcf: T")
    .Input("wco: T")
    .Input("b: T")
    .Output("i: T")
    .Output("cs: T")
    .Output("f: T")
    .Output("o: T")
    .Output("ci: T")
    .Output("co: T")
    .Output("h: T")
    .Attr("forget_bias: float = 1.0")
    .Attr("cell_clip: float = 3.0")
    .Attr("use_peephole: bool = false")
    .Attr("T: {half, float}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, cs_prev;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &cs_prev));

      DimensionHandle batch_size = c->Dim(x, 0);
      DimensionHandle cell_size = c->Dim(cs_prev, 1);
      ShapeHandle output = c->Matrix(batch_size, cell_size);
      for (int i = 0; i < 7; ++i) {
        c->set_output(i, output);
      }
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Computes the LSTM cell forward propagation for 1 time step.

This implementation uses 1 weight matrix and 1 bias vector, and there's an
optional peephole connection.

This kernel op implements the following mathematical equations:

```python
xh = [x, h_prev]
[i, f, ci, o] = xh * w + b
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

cell_clip: Value to clip the 'cs' value to.
use_peephole: Whether to use peephole weights.
forget_bias: The forget gate bias.

x: The input to the LSTM cell, shape (batch_size, num_inputs).
cs_prev: Value of the cell state at previous time step.
h_prev: Output of the previous cell at previous time step.
w: The weight matrix.
wci: The weight matrix for input gate peephole connection.
wcf: The weight matrix for forget gate peephole connection.
wco: The weight matrix for output gate peephole connection.
b: The bias vector.

i: The input gate.
cs: The cell state before the tanh.
f: The forget gate.
o: The output gate.
ci: The cell input.
co: The cell after the tanh.
h: The output h vector.
)doc");

REGISTER_OP("LSTMBlockCellGradOur")
    .Input("x: T")
    .Input("cs_prev: T")
    .Input("h_prev: T")
    .Input("w: T")
    .Input("wci: T")
    .Input("wcf: T")
    .Input("wco: T")
    .Input("b: T")
    .Input("i: T")
    .Input("cs: T")
    .Input("f: T")
    .Input("o: T")
    .Input("ci: T")
    .Input("co: T")
    .Input("cs_grad: T")
    .Input("h_grad: T")
    .Output("cs_prev_grad: T")
    .Output("dicfo: T")
    .Output("wci_grad: T")
    .Output("wcf_grad: T")
    .Output("wco_grad: T")
    .Attr("use_peephole: bool")
    .Attr("T: {half, float}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, cs_prev;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &cs_prev));

      DimensionHandle batch_size = c->Dim(x, 0);
      DimensionHandle cell_size = c->Dim(cs_prev, 1);
      DimensionHandle cell_size_times_4;
      TF_RETURN_IF_ERROR(c->Multiply(cell_size, 4, &cell_size_times_4));
      ShapeHandle cell_size_vec = c->Vector(cell_size);

      c->set_output(0, c->Matrix(batch_size, cell_size));
      c->set_output(1, c->Matrix(batch_size, cell_size_times_4));
      c->set_output(2, cell_size_vec);
      c->set_output(3, cell_size_vec);
      c->set_output(4, cell_size_vec);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Computes the LSTM cell backward propagation for 1 timestep.

This implementation is to be used in conjunction of LSTMBlockCell.

use_peephole: Whether the cell uses peephole connections.
x: The input to the LSTM cell, shape (batch_size, num_inputs).
cs_prev: The previous cell state.
h_prev: The previous h state.
w: The weight matrix.
wci: The weight matrix for input gate peephole connection.
wcf: The weight matrix for forget gate peephole connection.
wco: The weight matrix for output gate peephole connection.
b: The bias vector.
i: The input gate.
cs: The cell state before the tanh.
f: The forget gate.
o: The output gate.
ci: The cell input.
co: The cell after the tanh.
cs_grad: The current gradient of cs.
h_grad: The gradient of h vector.
cs_prev_grad: The gradient of cs to be back-propped.
dicfo: The derivative wrt to [i, cs, f, o].
wci_grad: The gradient for wci to be back-propped.
wcf_grad: The gradient for wcf to be back-propped.
wco_grad: The gradient for wco to be back-propped.
)doc");

}  // end namespace tensorflow



namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
void LSTMBlockCellFpropWithEigen(
    const LSTMBlockCell& cell, OpKernelContext* ctx, const CPUDevice& d,
    const float forget_bias, const float cell_clip, bool use_peephole,
    typename TTypes<T>::ConstMatrix x, typename TTypes<T>::ConstMatrix cs_prev,
    typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
    typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
    typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
    typename TTypes<T>::Matrix xh, typename TTypes<T>::Matrix i,
    typename TTypes<T>::Matrix cs, typename TTypes<T>::Matrix f,
    typename TTypes<T>::Matrix o, typename TTypes<T>::Matrix ci,
    typename TTypes<T>::Matrix co, typename TTypes<T>::Matrix icfo,
    typename TTypes<T>::Matrix h) {
  // Concat xh = [x, h].
  xh.slice(cell.xh_x_offsets(), cell.xh_x_extents()).device(d) = x;
  xh.slice(cell.xh_h_offsets(), cell.xh_h_extents()).device(d) = h_prev;

  // states1 = xh * w + b
  typename TTypes<T>::ConstMatrix const_xh(xh.data(), xh.dimensions());
  TensorBlasGemm<CPUDevice, T, false /* USE_CUBLAS */>::compute(
      ctx, d, false, false, typename gemm_compute_type<T>::type(1.f), const_xh,
      w, typename gemm_compute_type<T>::type(0.f), icfo);
  Eigen::array<Eigen::DenseIndex, 2> b_shape({1, b.dimensions()[0]});
  Eigen::array<Eigen::DenseIndex, 2> broadcast_shape({cell.batch_size(), 1});
  icfo.device(d) += b.reshape(b_shape).broadcast(broadcast_shape);

  Eigen::array<Eigen::DenseIndex, 2> p_shape({1, cell.cell_size()});
  Eigen::array<Eigen::DenseIndex, 2> p_broadcast_shape({cell.batch_size(), 1});

  // Input gate.
  if (use_peephole) {
    auto i_peep = cs_prev * wci.reshape(p_shape).broadcast(p_broadcast_shape);
    i.device(d) =
        (icfo.slice(cell.icfo_i_offsets(), cell.cell_extents()) + i_peep)
            .sigmoid();
  } else {
    i.device(d) =
        icfo.slice(cell.icfo_i_offsets(), cell.cell_extents()).sigmoid();
  }

  // Cell input.
  ci.device(d) = icfo.slice(cell.icfo_c_offsets(), cell.cell_extents()).tanh();

  // Forget gate (w/ bias).
  if (use_peephole) {
    auto f_peep = cs_prev * wcf.reshape(p_shape).broadcast(p_broadcast_shape);
    f.device(d) = (icfo.slice(cell.icfo_f_offsets(), cell.cell_extents()) +
                   f.constant(T(forget_bias)) + f_peep)
                      .sigmoid();
  } else {
    f.device(d) = (icfo.slice(cell.icfo_f_offsets(), cell.cell_extents()) +
                   f.constant(T(forget_bias)))
                      .sigmoid();
  }

  // cs = ci .* i + f .* cs_prev
  cs.device(d) = i * ci + f * cs_prev;

  if (cell_clip > 0.0f) {
    cs.device(d) =
        cs.binaryExpr(cs.constant(T(cell_clip)), Eigen::scalar_clip_op<T>());
  }

  // co = tanh(cs)
  co.device(d) = cs.tanh();

  // Output gate.
  if (use_peephole) {
    auto o_peep = cs * wco.reshape(p_shape).broadcast(p_broadcast_shape);
    o.device(d) =
        (icfo.slice(cell.icfo_o_offsets(), cell.cell_extents()) + o_peep)
            .sigmoid();
  } else {
    o.device(d) =
        icfo.slice(cell.icfo_o_offsets(), cell.cell_extents()).sigmoid();
  }

  // h = o .* co
  h.device(d) = o * co;
}

template <typename Device, typename T, bool USE_CUBLAS>
void LSTMBlockCellBpropWithEigen(
    const LSTMBlockCell& cell, OpKernelContext* ctx, const Device& d,
    bool use_peephole, typename TTypes<T>::ConstMatrix x,
    typename TTypes<T>::ConstMatrix cs_prev,
    typename TTypes<T>::ConstMatrix h_prev, typename TTypes<T>::ConstMatrix w,
    typename TTypes<T>::ConstVec wci, typename TTypes<T>::ConstVec wcf,
    typename TTypes<T>::ConstVec wco, typename TTypes<T>::ConstVec b,
    typename TTypes<T>::ConstMatrix i, typename TTypes<T>::ConstMatrix cs,
    typename TTypes<T>::ConstMatrix f, typename TTypes<T>::ConstMatrix o,
    typename TTypes<T>::ConstMatrix ci, typename TTypes<T>::ConstMatrix co,
    typename TTypes<T>::ConstMatrix cs_grad,
    typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,
    typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,
    typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,
    typename TTypes<T>::Matrix dicfo, typename TTypes<T>::Matrix cs_prev_grad,
    typename TTypes<T>::Vec wci_grad, typename TTypes<T>::Vec wcf_grad,
    typename TTypes<T>::Vec wco_grad) {

  make_sparse(di, 0.20);
  make_sparse(dci, 0.20);
  make_sparse(df, 0.20);
  make_sparse(do_, 0.20);

  // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
  do_.device(d) = o * (o.constant(T(1)) - o) * h_grad * co;

  // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t] + dcs[t + 1] .* f[t + 1]
  dcs.device(d) = (co.constant(T(1)) - co * co) * h_grad * o + cs_grad;

  Eigen::array<Eigen::DenseIndex, 2> p_shape({1, cell.cell_size()});
  Eigen::array<Eigen::DenseIndex, 2> p_broadcast_shape({cell.batch_size(), 1});
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

  dicfo.slice(cell.icfo_i_offsets(), cell.cell_extents()).device(d) = di;
  dicfo.slice(cell.icfo_c_offsets(), cell.cell_extents()).device(d) = dci;
  dicfo.slice(cell.icfo_f_offsets(), cell.cell_extents()).device(d) = df;
  dicfo.slice(cell.icfo_o_offsets(), cell.cell_extents()).device(d) = do_;

  cs_prev_grad.device(d) = dcs * f;
  if (use_peephole) {
    cs_prev_grad.device(d) =
        cs_prev_grad + di * wci.reshape(p_shape).broadcast(p_broadcast_shape) +
        df * wcf.reshape(p_shape).broadcast(p_broadcast_shape);
    wci_grad.device(d) = (di * cs_prev).sum(Eigen::array<int, 1>({0}));
    wcf_grad.device(d) = (df * cs_prev).sum(Eigen::array<int, 1>({0}));
    wco_grad.device(d) = (do_ * cs).sum(Eigen::array<int, 1>({0}));
  }
}

#define DEFINE_CPU_SPECS(T)                                                   \
  template <>                                                                 \
  void LSTMBlockCellFprop<CPUDevice, T, false /* USE_CUBLAS */>::operator()(  \
      OpKernelContext* ctx, const CPUDevice& d, const float forget_bias,      \
      const float cell_clip, bool use_peephole,                               \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstMatrix cs_prev,                                \
      typename TTypes<T>::ConstMatrix h_prev,                                 \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,    \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,     \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::Matrix xh,          \
      typename TTypes<T>::Matrix i, typename TTypes<T>::Matrix cs,            \
      typename TTypes<T>::Matrix f, typename TTypes<T>::Matrix o,             \
      typename TTypes<T>::Matrix ci, typename TTypes<T>::Matrix co,           \
      typename TTypes<T>::Matrix icfo, typename TTypes<T>::Matrix h) {        \
    LSTMBlockCellFpropWithEigen<T>(                                           \
        *this, ctx, d, forget_bias, cell_clip, use_peephole, x, cs_prev,      \
        h_prev, w, wci, wcf, wco, b, xh, i, cs, f, o, ci, co, icfo, h);       \
  }                                                                           \
  template <>                                                                 \
  void LSTMBlockCellBprop<CPUDevice, T, false /* USE_CUBLAS */>::operator()(  \
      OpKernelContext* ctx, const CPUDevice& d, bool use_peephole,            \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstMatrix cs_prev,                                \
      typename TTypes<T>::ConstMatrix h_prev,                                 \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,    \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,     \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::ConstMatrix i,      \
      typename TTypes<T>::ConstMatrix cs, typename TTypes<T>::ConstMatrix f,  \
      typename TTypes<T>::ConstMatrix o, typename TTypes<T>::ConstMatrix ci,  \
      typename TTypes<T>::ConstMatrix co,                                     \
      typename TTypes<T>::ConstMatrix cs_grad,                                \
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_, \
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,         \
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,           \
      typename TTypes<T>::Matrix dicfo,                                       \
      typename TTypes<T>::Matrix cs_prev_grad,                                \
      typename TTypes<T>::Vec wci_grad, typename TTypes<T>::Vec wcf_grad,     \
      typename TTypes<T>::Vec wco_grad) {                                     \
    LSTMBlockCellBpropWithEigen<CPUDevice, T, false /* USE_CUBLAS */>(        \
        *this, ctx, d, use_peephole, x, cs_prev, h_prev, w, wci, wcf, wco, b, \
        i, cs, f, o, ci, co, cs_grad, h_grad, do_, dcs, dci, df, di, dicfo,   \
        cs_prev_grad, wci_grad, wcf_grad, wco_grad);                          \
  }                                                                           \
  template struct LSTMBlockCellFprop<CPUDevice, T, false /* USE_CUBLAS */>;   \
  template struct LSTMBlockCellBprop<CPUDevice, T, false /* USE_CUBLAS */>;

DEFINE_CPU_SPECS(float);
DEFINE_CPU_SPECS(Eigen::half);
#undef DEFINE_CPU_SPECS

}  // namespace functor

template <typename Device, typename T, bool USE_CUBLAS>
class LSTMBlockCellOp : public OpKernel {
 public:
  explicit LSTMBlockCellOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_clip", &cell_clip_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
  }

  void Compute(OpKernelContext* ctx) override {

    //std::cout << "No czesc tutaj forward prop\n";

    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 cell_size = cs_prev_tensor->dim_size(1);

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                                        cs_prev_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != input_size + cell_size: ",
                    w_tensor->dim_size(0), " vs. ", input_size + cell_size));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", w_tensor->dim_size(1),
                    " vs. ", cell_size * 4));

    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", b_tensor->dim_size(0),
                    " vs. ", cell_size * 4));

    // Allocate our output tensors.
    Tensor* i_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {"h_prev"}, "i",
                            TensorShape({batch_size, cell_size}), &i_tensor));

    Tensor* cs_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("cs", TensorShape({batch_size, cell_size}),
                                  &cs_tensor));

    Tensor* f_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("f", TensorShape({batch_size, cell_size}),
                                  &f_tensor));

    Tensor* o_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {"cs_prev"}, "o",
                            TensorShape({batch_size, cell_size}), &o_tensor));

    Tensor* ci_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("ci", TensorShape({batch_size, cell_size}),
                                  &ci_tensor));

    Tensor* co_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("co", TensorShape({batch_size, cell_size}),
                                  &co_tensor));

    Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("h", TensorShape({batch_size, cell_size}),
                                  &h_tensor));

    // Allocate our temp tensors.
    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::v(),
                            TensorShape({batch_size, input_size + cell_size}),
                            &xh_tensor));

    Tensor icfo_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                      TensorShape({batch_size, cell_size * 4}),
                                      &icfo_tensor));

    const Device& device = ctx->eigen_device<Device>();

    functor::LSTMBlockCellFprop<Device, T, USE_CUBLAS>(batch_size, input_size,
                                                       cell_size)(
        ctx, device, forget_bias_, cell_clip_, use_peephole_,
        x_tensor->matrix<T>(), cs_prev_tensor->matrix<T>(),
        h_prev_tensor->matrix<T>(), w_tensor->matrix<T>(), wci_tensor->vec<T>(),
        wcf_tensor->vec<T>(), wco_tensor->vec<T>(), b_tensor->vec<T>(),
        xh_tensor.matrix<T>(), i_tensor->matrix<T>(), cs_tensor->matrix<T>(),
        f_tensor->matrix<T>(), o_tensor->matrix<T>(), ci_tensor->matrix<T>(),
        co_tensor->matrix<T>(), icfo_tensor.matrix<T>(), h_tensor->matrix<T>());
  }

 private:
  float forget_bias_;
  float cell_clip_;
  bool use_peephole_;
};

#define REGISTER_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("LSTMBlockCellOur").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LSTMBlockCellOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
REGISTER_KERNEL(Eigen::half);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                \
  template <>                                                              \
  void LSTMBlockCellFprop<GPUDevice, T, true>::operator()(                 \
      OpKernelContext* ctx, const GPUDevice& d, const float forget_bias,   \
      const float cell_clip, bool use_peephole,                            \
      typename TTypes<T>::ConstMatrix x,                                   \
      typename TTypes<T>::ConstMatrix cs_prev,                             \
      typename TTypes<T>::ConstMatrix h_prev,                              \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci, \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,  \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::Matrix xh,       \
      typename TTypes<T>::Matrix i, typename TTypes<T>::Matrix cs,         \
      typename TTypes<T>::Matrix f, typename TTypes<T>::Matrix o,          \
      typename TTypes<T>::Matrix ci, typename TTypes<T>::Matrix co,        \
      typename TTypes<T>::Matrix icfo, typename TTypes<T>::Matrix h);      \
                                                                           \
  extern template struct LSTMBlockCellFprop<GPUDevice, T, true>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
#undef DECLARE_GPU_SPEC
}  // end namespace functor

#define REGISTER_GPU_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("LSTMBlockCell").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LSTMBlockCellOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(Eigen::half);
// REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

template <typename Device, typename T, bool USE_CUBLAS>
class LSTMBlockCellGradOp : public OpKernel {
 public:
  explicit LSTMBlockCellGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const Tensor* i_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("i", &i_tensor));

    const Tensor* cs_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs", &cs_tensor));

    const Tensor* f_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("f", &f_tensor));

    const Tensor* o_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("o", &o_tensor));

    const Tensor* ci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ci", &ci_tensor));

    const Tensor* co_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("co", &co_tensor));

    const Tensor* cs_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_grad", &cs_grad_tensor));

    const Tensor* h_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_grad", &h_grad_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);
    const int64 cell_size = cs_prev_tensor->dim_size(1);

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                                        cs_prev_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w.dim_size(0) != input_size + cell_size: ",
                    w_tensor->dim_size(0), " vs. ", input_size + cell_size));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", w_tensor->dim_size(1),
                    " vs. ", cell_size * 4));

    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", b_tensor->dim_size(0),
                    " vs. ", cell_size * 4));

    OP_REQUIRES(ctx, i_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "i.dim_size(0) != batch_size: ", i_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, i_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "i.dim_size(1) != cell_size: ", i_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, cs_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "cs.dim_size(0) != batch_size: ", cs_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, cs_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "cs.dim_size(1) != cell_size: ", cs_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, f_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "f.dim_size(0) != batch_size: ", f_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, f_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "i.dim_size(1) != cell_size: ", f_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, o_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "o.dim_size(0) != batch_size: ", o_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, o_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "o.dim_size(1) != cell_size: ", o_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, ci_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "ci.dim_size(0) != batch_size: ", ci_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, ci_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "ci.dim_size(1) != cell_size: ", ci_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, co_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "co.dim_size(0) != batch_size: ", co_tensor->dim_size(0),
                    " vs. ", batch_size));
    OP_REQUIRES(ctx, co_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "co.dim_size(1) != cell_size: ", co_tensor->dim_size(1),
                    " vs. ", cell_size));

    OP_REQUIRES(ctx, cs_grad_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "cs_grad_tensor.dims(0) != batch_size: ",
                    cs_grad_tensor->dim_size(0), " vs. ", batch_size));
    OP_REQUIRES(ctx, cs_grad_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("cs_grad_tensor.dims(1) != cell_size: ",
                                        cs_grad_tensor->dim_size(1), " vs. ",
                                        cell_size));

    OP_REQUIRES(ctx, h_grad_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_grad_tensor.dims(0) != batch_size: ",
                                        h_grad_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_grad_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument("h_grad_tensor.dims(1) != cell_size: ",
                                        h_grad_tensor->dim_size(1), " vs. ",
                                        cell_size));

    // Allocate our output tensors.
    Tensor* cs_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output(
                 {"cs_grad"}, "cs_prev_grad",
                 TensorShape({batch_size, cell_size}), &cs_prev_grad_tensor));

    Tensor* dicfo_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            "dicfo", TensorShape({batch_size, cell_size * 4}),
                            &dicfo_tensor));

    Tensor* wci_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output(
                 {"wci"}, "wci_grad", wci_tensor->shape(), &wci_grad_tensor));

    Tensor* wcf_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output(
                 {"wcf"}, "wcf_grad", wcf_tensor->shape(), &wcf_grad_tensor));

    Tensor* wco_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output(
                 {"wco"}, "wco_grad", wco_tensor->shape(), &wco_grad_tensor));

    // Allocate our temp tensors.
    Tensor do_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &do_tensor));

    Tensor dcs_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &dcs_tensor));

    Tensor dci_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &dci_tensor));

    Tensor df_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &df_tensor));

    Tensor di_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                           TensorShape({batch_size, cell_size}),
                                           &di_tensor));

    const Device& device = ctx->eigen_device<Device>();

    functor::TensorZero<Device, T>()(device, wci_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, wcf_grad_tensor->flat<T>());
    functor::TensorZero<Device, T>()(device, wco_grad_tensor->flat<T>());

    functor::LSTMBlockCellBprop<Device, T, USE_CUBLAS>(batch_size, input_size,
                                                       cell_size)(
        ctx, device, use_peephole_, x_tensor->matrix<T>(),
        cs_prev_tensor->matrix<T>(), h_prev_tensor->matrix<T>(),
        w_tensor->matrix<T>(), wci_tensor->vec<T>(), wcf_tensor->vec<T>(),
        wco_tensor->vec<T>(), b_tensor->vec<T>(), i_tensor->matrix<T>(),
        cs_tensor->matrix<T>(), f_tensor->matrix<T>(), o_tensor->matrix<T>(),
        ci_tensor->matrix<T>(), co_tensor->matrix<T>(),
        cs_grad_tensor->matrix<T>(), h_grad_tensor->matrix<T>(),
        do_tensor.matrix<T>(), dcs_tensor.matrix<T>(), dci_tensor.matrix<T>(),
        df_tensor.matrix<T>(), di_tensor.matrix<T>(), dicfo_tensor->matrix<T>(),
        cs_prev_grad_tensor->matrix<T>(), wci_grad_tensor->vec<T>(),
        wcf_grad_tensor->vec<T>(), wco_grad_tensor->vec<T>());
  }

 protected:
  bool use_peephole_;
};

#define REGISTER_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("LSTMBlockCellGradOur").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LSTMBlockCellGradOp<CPUDevice, T, false>);
REGISTER_KERNEL(float);
REGISTER_KERNEL(Eigen::half);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                   \
  template <>                                                                 \
  void LSTMBlockCellBprop<GPUDevice, T, true>::operator()(                    \
      OpKernelContext* ctx, const GPUDevice& d, bool use_peephole,            \
      typename TTypes<T>::ConstMatrix x,                                      \
      typename TTypes<T>::ConstMatrix cs_prev,                                \
      typename TTypes<T>::ConstMatrix h_prev,                                 \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,    \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,     \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::ConstMatrix i,      \
      typename TTypes<T>::ConstMatrix cs, typename TTypes<T>::ConstMatrix f,  \
      typename TTypes<T>::ConstMatrix o, typename TTypes<T>::ConstMatrix ci,  \
      typename TTypes<T>::ConstMatrix co,                                     \
      typename TTypes<T>::ConstMatrix cs_grad,                                \
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_, \
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,         \
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,           \
      typename TTypes<T>::Matrix dicfo,                                       \
      typename TTypes<T>::Matrix cs_prev_grad,                                \
      typename TTypes<T>::Vec wci_grad, typename TTypes<T>::Vec wcf_grad,     \
      typename TTypes<T>::Vec wco_grad);                                      \
                                                                              \
  extern template struct LSTMBlockCellBprop<GPUDevice, T,                     \
                                            true /* USE_CUBLAS */>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
// DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("LSTMBlockCellGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LSTMBlockCellGradOp<GPUDevice, T, true>);

REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(Eigen::half);
// REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
