// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/fast_rcnn_layers.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {
#ifdef USE_GREENTEA
	const char* const forward_layer =
		"\n"
		"\n"
		"__kernel void SmoothL1Forward(const int n, __global const Dtype* in, __global Dtype* out,  Dtype sigma2) {\n"
		"  // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma\n"
		"  //        |x| - 0.5 / sigma / sigma    otherwise\n"
		"  for (int index = get_global_id(0); index < n; index += get_global_id(0)) {\n"
		"    Dtype val = in[index];\n"
		"    Dtype abs_val = fabs(val);\n"
		"    if (abs_val < 1.0 / sigma2) {\n"
		"      out[index] = 0.5 * val * val * sigma2;\n"
		"    } else {\n"
		"      out[index] = abs_val - 0.5 / sigma2;\n"
		"    }\n"
		"  }\n"
		"}\n"
		"\n";
const char* const backwards_layer=
		"\n"
		"__kernel void SmoothL1Backward(const int n, __global const Dtype* in, __global Dtype* out,\n"
		"    Dtype sigma2) {\n"
		"  // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma\n"
		"  //       = sign(x)                   otherwise\n"
		"  for (int index = get_global_id(0); index < n; index += get_global_id(0)) {\n"
		"    Dtype val = in[index];\n"
		"    Dtype abs_val = fabs(val);\n"
		"    if (abs_val < 1.0 / sigma2) {\n"
		"      out[index] = sigma2 * val;\n"
		"    } else {\n"
		"      out[index] = ((Dtype)(0) < val) - (val < (Dtype)(0));\n"
		"    }\n"
		"  }\n"
		"}\n"
		"\n"
		"\n";


template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	viennacl::ocl::context &ctx = viennacl::ocl::get_context(
		this->device_->id());
	static bool compiled = false;
	if (!compiled)
	{
		std::string kernel;
		if (is_same<Dtype, float>::value)
			kernel = "#define DType float\n";
		else if (is_same<Dtype, double>::value)
			kernel = "#define DType double\n";
		kernel += forward_layer;
		ctx.add_program(kernel.c_str(), CL_KERNEL_SELECT("SmoothL1Forward"));
		compiled = true;
	}
	static viennacl::ocl::program &program = ctx.get_program(CL_KERNEL_SELECT("SmoothL1Forward"));
	static viennacl::ocl::kernel& forward_pool = program.get_kernel("SmoothL1Forward");
	forward_pool.global_work_size(256 * 64);

	int count = bottom[0]->count();
	greentea_gpu_sub<Dtype>(this->device_->id(),
		count,
		(const cl_mem)bottom[0]->gpu_data(),0,
		(const cl_mem)bottom[1]->gpu_data(),0,
		(cl_mem)diff_.mutable_gpu_data(),0);    // d := b0 - b1
	if (has_weights_) {
		// apply "inside" weights
		greentea_gpu_mul<Dtype>(this->device_->id(),
			count,
			(const cl_mem)bottom[2]->gpu_data(),0,
			(const cl_mem)diff_.gpu_data(),0,
			(cl_mem)diff_.mutable_gpu_data(),0);  // d := w_in * (b0 - b1)
	}
	viennacl::ocl::enqueue(forward_pool(
		count, WrapHandle((cl_mem)diff_.gpu_data(), &ctx), WrapHandle((cl_mem)errors_.mutable_gpu_data(), &ctx), sigma2_),
		ctx.get_queue());

	if (has_weights_) {
		// apply "outside" weights
		greentea_gpu_mul<Dtype>(this->device_->id(),
			count,
			(cl_mem)bottom[3]->gpu_data(),0,
			(cl_mem)errors_.gpu_data(), 0,
			(cl_mem)errors_.mutable_gpu_data(),0);  // d := w_out * SmoothL1(w_in * (b0 - b1))
	}

	Dtype loss;
	greentea_gpu_dot<Dtype>(this->device_->id(), count, (cl_mem)ones_.gpu_data(),0,(cl_mem) errors_.gpu_data(),0, &loss);
	top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}


template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// after forwards, diff_ holds w_in * (b0 - b1)
	viennacl::ocl::context &ctx = viennacl::ocl::get_context(this->device_->id());
	static bool compiled = false;
	if (!compiled)
	{
		std::string kernel;
		if (is_same<Dtype, float>::value)
			kernel = "#define DType float\n";
		else if (is_same<Dtype, double>::value)
			kernel = "#define DType double\n";
		kernel += backwards_layer;
		ctx.add_program(kernel.c_str(), CL_KERNEL_SELECT("SmoothL1Backward"));
		compiled = true;
	}
	static viennacl::ocl::program &program = ctx.get_program(CL_KERNEL_SELECT("SmoothL1Backward"));
	static viennacl::ocl::kernel& back_pool = program.get_kernel("SmoothL1Backward");
	back_pool.global_work_size(256 * 64);

	int count = diff_.count();
	viennacl::ocl::enqueue(
		back_pool(
			count, WrapHandle((cl_mem)diff_.gpu_data(), &ctx), WrapHandle((cl_mem)diff_.mutable_gpu_data(), &ctx), sigma2_
		), ctx.get_queue());
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
			greentea_gpu_axpby<Dtype>(this->device_->id(),
				count,                           // count
				alpha,                           // alpha
				(const cl_mem)diff_.gpu_data(),0,                // x
				Dtype(0),                        // beta
				(cl_mem)bottom[i]->mutable_gpu_diff(),0);  // y
			if (has_weights_) {
				// Scale by "inside" weight
				greentea_gpu_mul<Dtype>(this->device_->id(),
					count,
					(const cl_mem)bottom[2]->gpu_data(),0,
					(const cl_mem)bottom[i]->gpu_diff(),0,
					(cl_mem)bottom[i]->mutable_gpu_diff(),0);
				// Scale by "outside" weight
				greentea_gpu_mul<Dtype>(this->device_->id(),
					count,
					(const cl_mem)bottom[3]->gpu_data(),0,
					(const cl_mem)bottom[i]->gpu_diff(),0,
					(cl_mem)bottom[i]->mutable_gpu_diff(),0);
			}
		}
	}
}

#endif

#ifdef USE_CUDA
template <typename Dtype>
__global__ void SmoothL1Forward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
  //        |x| - 0.5 / sigma / sigma    otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = 0.5 * val * val * sigma2;
    } else {
      out[index] = abs_val - 0.5 / sigma2;
    }
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());    // d := b0 - b1
  if (has_weights_) {
    // apply "inside" weights
    caffe_gpu_mul(
        count,
        bottom[2]->gpu_data(),
        diff_.gpu_data(),
        diff_.mutable_gpu_data());  // d := w_in * (b0 - b1)
  }
  SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), errors_.mutable_gpu_data(), sigma2_);
  CUDA_POST_KERNEL_CHECK;

  if (has_weights_) {
    // apply "outside" weights
    caffe_gpu_mul(
        count,
        bottom[3]->gpu_data(),
        errors_.gpu_data(),
        errors_.mutable_gpu_data());  // d := w_out * SmoothL1(w_in * (b0 - b1))
  }

  Dtype loss;
  caffe_gpu_dot(count, ones_.gpu_data(), errors_.gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
__global__ void SmoothL1Backward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
  //       = sign(x)                   otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = sigma2 * val;
    } else {
      out[index] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // after forwards, diff_ holds w_in * (b0 - b1)
  int count = diff_.count();
  SmoothL1Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), diff_.mutable_gpu_data(), sigma2_);
  CUDA_POST_KERNEL_CHECK;
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          count,                           // count
          alpha,                           // alpha
          diff_.gpu_data(),                // x
          Dtype(0),                        // beta
          bottom[i]->mutable_gpu_diff());  // y
      if (has_weights_) {
        // Scale by "inside" weight
        caffe_gpu_mul(
            count,
            bottom[2]->gpu_data(),
            bottom[i]->gpu_diff(),
            bottom[i]->mutable_gpu_diff());
        // Scale by "outside" weight
        caffe_gpu_mul(
            count,
            bottom[3]->gpu_data(),
            bottom[i]->gpu_diff(),
            bottom[i]->mutable_gpu_diff());
      }
    }
  }
}
#endif

INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1LossLayer);

}  // namespace caffe
