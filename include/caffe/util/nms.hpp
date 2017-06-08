#ifndef _CAFFE_UTIL_NMS_HPP_
#define _CAFFE_UTIL_NMS_HPP_

#include <vector>

#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
void nms_cpu(const int num_boxes,
             const Dtype boxes[],
             int index_out[],
             int* const num_out,
             const int base_index,
             const Dtype nms_thresh,
             const int max_num_out);
#ifdef USE_CUDA
template <typename Dtype>
void nms_gpu( const int num_boxes,
             const Dtype boxes_gpu[],
             Blob<int>* const p_mask,
             int index_out_cpu[],
             int* const num_out,
             const int base_index,
             const Dtype nms_thresh,
             const int max_num_out);
#endif
#ifdef USE_GREENTEA
template <typename Dtype>
void nms_gpu(
	const int device_id,
	const int num_boxes,
	const Dtype boxes_gpu[],
	Blob<int>* const p_mask,
	int index_out_cpu[],
	int* const num_out,
	const int base_index,
	const Dtype nms_thresh,
	const int max_num_out);
#endif
}  // namespace caffe

#endif  // CAFFE_UTIL_NMS_HPP_
