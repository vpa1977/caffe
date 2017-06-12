#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/util/nms.hpp"


#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_GREENTEA
const char* const proposal_kernel =
"\n"
"#define Dtype float\n"
"\n"
"static int transform_box(__global Dtype* box,\n"
"                  const Dtype dx, const Dtype dy,\n"
"                  const Dtype d_log_w, const Dtype d_log_h,\n"
"                  const Dtype img_W, const Dtype img_H,\n"
"                  const Dtype min_box_W, const Dtype min_box_H)\n"
"{\n"
"  // width & height of box\n"
"  const Dtype w = box[2] - box[0] + (Dtype)1;\n"
"  const Dtype h = box[3] - box[1] + (Dtype)1;\n"
"  // center location of box\n"
"  const Dtype ctr_x = box[0] + (Dtype)0.5 * w;\n"
"  const Dtype ctr_y = box[1] + (Dtype)0.5 * h;\n"
"\n"
"  // new center location according to gradient (dx, dy)\n"
"  const Dtype pred_ctr_x = dx * w + ctr_x;\n"
"  const Dtype pred_ctr_y = dy * h + ctr_y;\n"
"  // new width & height according to gradient d(log w), d(log h)\n"
"  const Dtype pred_w = exp(d_log_w) * w;\n"
"  const Dtype pred_h = exp(d_log_h) * h;\n"
"\n"
"  // update upper-left corner location\n"
"  box[0] = pred_ctr_x - (Dtype)0.5 * pred_w;\n"
"  box[1] = pred_ctr_y - (Dtype)0.5 * pred_h;\n"
"  // update lower-right corner location\n"
"  box[2] = pred_ctr_x + (Dtype)0.5 * pred_w;\n"
"  box[3] = pred_ctr_y + (Dtype)0.5 * pred_h;\n"
"\n"
"  // adjust new corner locations to be within the image region,\n"
"  box[0] = max((Dtype)0,  min(box[0],  img_W - (Dtype)1));\n"
"  box[1] = max((Dtype)0,  min(box[1],  img_H - (Dtype)1));\n"
"  box[2] = max((Dtype)0,  min(box[2],  img_W - (Dtype)1));\n"
"  box[3] = max((Dtype)0,  min(box[3],  img_H - (Dtype)1));\n"
"\n"
"  // recompute new width & height\n"
"  const Dtype box_w = box[2] - box[0] + (Dtype)1;\n"
"  const Dtype box_h = box[3] - box[1] + (Dtype)1;\n"
"\n"
"  // check if new box's size >= threshold\n"
"  return (box_w >= min_box_W) * (box_h >= min_box_H);\n"
"}\n"
"\n"
"static void sort_box(__global Dtype* list_cpu, const int start, const int end,\n"
"              const int num_top)\n"
"{\n"
"  const Dtype pivot_score = list_cpu[start * 5 + 4];\n"
"  int left = start + 1, right = end;\n"
"  Dtype temp[5];\n"
"  while (left <= right) {\n"
"    while (left <= end && list_cpu[left * 5 + 4] >= pivot_score) ++left;\n"
"    while (right > start && list_cpu[right * 5 + 4] <= pivot_score) --right;\n"
"    if (left <= right) {\n"
"      for (int i = 0; i < 5; ++i) {\n"
"        temp[i] = list_cpu[left * 5 + i];\n"
"      }\n"
"      for (int i = 0; i < 5; ++i) {\n"
"        list_cpu[left * 5 + i] = list_cpu[right * 5 + i];\n"
"      }\n"
"      for (int i = 0; i < 5; ++i) {\n"
"        list_cpu[right * 5 + i] = temp[i];\n"
"      }\n"
"      ++left;\n"
"      --right;\n"
"    }\n"
"  }\n"
"\n"
"  if (right > start) {\n"
"    for (int i = 0; i < 5; ++i) {\n"
"      temp[i] = list_cpu[start * 5 + i];\n"
"    }\n"
"    for (int i = 0; i < 5; ++i) {\n"
"      list_cpu[start * 5 + i] = list_cpu[right * 5 + i];\n"
"    }\n"
"    for (int i = 0; i < 5; ++i) {\n"
"      list_cpu[right * 5 + i] = temp[i];\n"
"    }\n"
"  }\n"
"\n"
"  if (start < right - 1) {\n"
"    sort_box(list_cpu, start, right - 1, num_top);\n"
"  }\n"
"  if (right + 1 < num_top && right + 1 < end) {\n"
"    sort_box(list_cpu, right + 1, end, num_top);\n"
"  }\n"
"}\n"
"\n"
"__kernel void enumerate_proposals_gpu(const int nthreads,\n"
"                             __global const Dtype* bottom4d_in, \n"
"                             __global const Dtype* d_anchor4d,\n"
"                             __global const Dtype* anchors,\n"
"                             __global Dtype* proposals,\n"
"                             const int num_anchors,\n"
"                             const int bottom_H, const int bottom_W,\n"
"                             const Dtype img_H, const Dtype img_W,\n"
"                             const Dtype min_box_H, const Dtype min_box_W,\n"
"                             const int feat_stride)\n"
"{\n"
"	 __global const Dtype* bottom4d = bottom4d_in + nthreads;\n"
"	for (int index = get_global_id(0); index < nthreads; index += get_global_size(0))\n"
"  {\n"
"    const int h = index / num_anchors / bottom_W;\n"
"    const int w = (index / num_anchors) % bottom_W;\n"
"    const int k = index % num_anchors;\n"
"    const Dtype x = w * feat_stride;\n"
"    const Dtype y = h * feat_stride;\n"
"    __global const Dtype* p_box = d_anchor4d + h * bottom_W + w;\n"
"    __global const Dtype* p_score = bottom4d + h * bottom_W + w;\n"
"\n"
"    const int bottom_area = bottom_H * bottom_W;\n"
"    const Dtype dx = p_box[(k * 4 + 0) * bottom_area];\n"
"    const Dtype dy = p_box[(k * 4 + 1) * bottom_area];\n"
"    const Dtype d_log_w = p_box[(k * 4 + 2) * bottom_area];\n"
"    const Dtype d_log_h = p_box[(k * 4 + 3) * bottom_area];\n"
"\n"
"		__global Dtype* const p_proposal = proposals + index * 5;\n"
"    p_proposal[0] = x + anchors[k * 4 + 0];\n"
"    p_proposal[1] = y + anchors[k * 4 + 1];\n"
"    p_proposal[2] = x + anchors[k * 4 + 2];\n"
"    p_proposal[3] = y + anchors[k * 4 + 3];\n"
"    p_proposal[4]\n"
"        = transform_box(p_proposal,\n"
"                        dx, dy, d_log_w, d_log_h,\n"
"                        img_W, img_H, min_box_W, min_box_H)\n"
"          * p_score[k * bottom_area];\n"
"  }\n"
"}\n"
"\n"
"__kernel \n"
"void retrieve_rois_gpu(const int nthreads,\n"
"                       const int item_index,\n"
"                       __global const Dtype* proposals,\n"
"                       __global const int* roi_indices,\n"
"                       __global Dtype* rois,\n"
"                       __global Dtype* roi_scores)\n"
"{\n"
"  for (int index = get_global_id(0); index < nthreads; index += get_global_size(0))\n"
"	{\n"
"    __global const Dtype* const proposals_index = proposals + roi_indices[index] * 5;\n"
"    rois[index * 5 + 0] = item_index;\n"
"    rois[index * 5 + 1] = proposals_index[0];\n"
"    rois[index * 5 + 2] = proposals_index[1];\n"
"    rois[index * 5 + 3] = proposals_index[2];\n"
"    rois[index * 5 + 4] = proposals_index[3];\n"
"    if (roi_scores) {\n"
"      roi_scores[index] = proposals_index[4];\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"\n";


template <typename Dtype>
static
void sort_box(Dtype list_cpu[], const int start, const int end,
	const int num_top)
{
	const Dtype pivot_score = list_cpu[start * 5 + 4];
	int left = start + 1, right = end;
	Dtype temp[5];
	while (left <= right) {
		while (left <= end && list_cpu[left * 5 + 4] >= pivot_score) ++left;
		while (right > start && list_cpu[right * 5 + 4] <= pivot_score) --right;
		if (left <= right) {
			for (int i = 0; i < 5; ++i) {
				temp[i] = list_cpu[left * 5 + i];
			}
			for (int i = 0; i < 5; ++i) {
				list_cpu[left * 5 + i] = list_cpu[right * 5 + i];
			}
			for (int i = 0; i < 5; ++i) {
				list_cpu[right * 5 + i] = temp[i];
			}
			++left;
			--right;
		}
	}

	if (right > start) {
		for (int i = 0; i < 5; ++i) {
			temp[i] = list_cpu[start * 5 + i];
		}
		for (int i = 0; i < 5; ++i) {
			list_cpu[start * 5 + i] = list_cpu[right * 5 + i];
		}
		for (int i = 0; i < 5; ++i) {
			list_cpu[right * 5 + i] = temp[i];
		}
	}

	if (start < right - 1) {
		sort_box(list_cpu, start, right - 1, num_top);
	}
	if (right + 1 < num_top && right + 1 < end) {
		sort_box(list_cpu, right + 1, end, num_top);
	}
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	CHECK_EQ(bottom[0]->shape(0), 1) << "Only single item batches are supported";
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
		kernel += proposal_kernel;
		ctx.add_program(kernel.c_str(), CL_KERNEL_SELECT("proposal"));
		compiled = true;
	}
	static viennacl::ocl::program &program = ctx.get_program(CL_KERNEL_SELECT("proposal"));
	static viennacl::ocl::kernel &enumerate_kernel = program.get_kernel("enumerate_proposals_gpu");
	static viennacl::ocl::kernel &retrieve_rois_kernel = program.get_kernel("retrieve_rois_gpu");
	enumerate_kernel.global_work_size(256 * 64);
	retrieve_rois_kernel.global_work_size(256 * 64);

	viennacl::ocl::handle<cl_mem> p_bottom_item = WrapHandle((cl_mem)bottom[0]->gpu_data(), &ctx);
	viennacl::ocl::handle<cl_mem> p_d_anchor_item = WrapHandle((cl_mem)bottom[1]->gpu_data(), &ctx);
	const Dtype* p_img_info_cpu = bottom[2]->cpu_data();
	viennacl::ocl::handle<cl_mem> p_roi_item = WrapHandle((cl_mem)top[0]->mutable_gpu_data(), &ctx);
	viennacl::ocl::handle<cl_mem> p_score_item = (top.size() > 1) ? WrapHandle((cl_mem)top[1]->mutable_gpu_data(), &ctx) : WrapHandle((cl_mem)0, &ctx);

	vector<int> proposals_shape(2);
	vector<int> top_shape(2);
	proposals_shape[0] = 0;
	proposals_shape[1] = 5;
	top_shape[0] = 0;
	top_shape[1] = 5;

	for (int n = 0; n < bottom[0]->shape(0); ++n) {
		// bottom shape: (2 x num_anchors) x H x W
		const int bottom_H = bottom[0]->height();
		const int bottom_W = bottom[0]->width();
		// input image height & width
		const Dtype img_H = p_img_info_cpu[0];
		const Dtype img_W = p_img_info_cpu[1];
		// scale factor for height & width
		const Dtype scale_H = p_img_info_cpu[2];
		const Dtype scale_W = p_img_info_cpu[3];
		// minimum box width & height
		const Dtype min_box_H = min_size_ * scale_H;
		const Dtype min_box_W = min_size_ * scale_W;
		// number of all proposals = num_anchors * H * W
		const int num_proposals = anchors_.shape(0) * bottom_H * bottom_W;
		// number of top-n proposals before NMS
		const int pre_nms_topn = std::min(num_proposals, pre_nms_topn_);
		// number of final RoIs
		int num_rois = 0;

		// enumerate all proposals
		//   num_proposals = num_anchors * H * W
		//   (x1, y1, x2, y2, score) for each proposal
		// NOTE: for bottom, only foreground scores are passed
		proposals_shape[0] = num_proposals;
		proposals_.Reshape(proposals_shape);
		viennacl::ocl::enqueue(
			enumerate_kernel(
				num_proposals,
				p_bottom_item, p_d_anchor_item,
				WrapHandle((cl_mem)anchors_.gpu_data(), &ctx),
				WrapHandle((cl_mem)proposals_.mutable_gpu_data(), &ctx),
				anchors_.shape(0),
				bottom_H, bottom_W, img_H, img_W, min_box_H, min_box_W,
				feat_stride_),
			ctx.get_queue());

		sort_box(proposals_.mutable_cpu_data(), 0, num_proposals - 1, pre_nms_topn_);

		nms_gpu(this->device_->id(), pre_nms_topn,
			proposals_.gpu_data(), 
			&nms_mask_,
			roi_indices_.mutable_cpu_data(), &num_rois,
			0, nms_thresh_, post_nms_topn_);

		viennacl::ocl::enqueue(
			retrieve_rois_kernel(
				num_rois, n,
				WrapHandle((cl_mem)proposals_.gpu_data(), &ctx),
				WrapHandle((cl_mem)roi_indices_.gpu_data(), &ctx),
				p_roi_item, p_score_item),
			ctx.get_queue());

		top_shape[0] += num_rois;
	}

	top[0]->Reshape(top_shape);
	if (top.size() > 1) {
		top_shape.pop_back();
		top[1]->Reshape(top_shape);
	}
}


#endif 

#ifdef USE_CUDA
template <typename Dtype>
__device__
static
int transform_box(Dtype box[],
                  const Dtype dx, const Dtype dy,
                  const Dtype d_log_w, const Dtype d_log_h,
                  const Dtype img_W, const Dtype img_H,
                  const Dtype min_box_W, const Dtype min_box_H)
{
  // width & height of box
  const Dtype w = box[2] - box[0] + (Dtype)1;
  const Dtype h = box[3] - box[1] + (Dtype)1;
  // center location of box
  const Dtype ctr_x = box[0] + (Dtype)0.5 * w;
  const Dtype ctr_y = box[1] + (Dtype)0.5 * h;

  // new center location according to gradient (dx, dy)
  const Dtype pred_ctr_x = dx * w + ctr_x;
  const Dtype pred_ctr_y = dy * h + ctr_y;
  // new width & height according to gradient d(log w), d(log h)
  const Dtype pred_w = exp(d_log_w) * w;
  const Dtype pred_h = exp(d_log_h) * h;

  // update upper-left corner location
  box[0] = pred_ctr_x - (Dtype)0.5 * pred_w;
  box[1] = pred_ctr_y - (Dtype)0.5 * pred_h;
  // update lower-right corner location
  box[2] = pred_ctr_x + (Dtype)0.5 * pred_w;
  box[3] = pred_ctr_y + (Dtype)0.5 * pred_h;

  // adjust new corner locations to be within the image region,
  box[0] = max((Dtype)0,  min(box[0],  img_W - (Dtype)1));
  box[1] = max((Dtype)0,  min(box[1],  img_H - (Dtype)1));
  box[2] = max((Dtype)0,  min(box[2],  img_W - (Dtype)1));
  box[3] = max((Dtype)0,  min(box[3],  img_H - (Dtype)1));

  // recompute new width & height
  const Dtype box_w = box[2] - box[0] + (Dtype)1;
  const Dtype box_h = box[3] - box[1] + (Dtype)1;

  // check if new box's size >= threshold
  return (box_w >= min_box_W) * (box_h >= min_box_H);
}

template <typename Dtype>
static
void sort_box(Dtype list_cpu[], const int start, const int end,
              const int num_top)
{
  const Dtype pivot_score = list_cpu[start * 5 + 4];
  int left = start + 1, right = end;
  Dtype temp[5];
  while (left <= right) {
    while (left <= end && list_cpu[left * 5 + 4] >= pivot_score) ++left;
    while (right > start && list_cpu[right * 5 + 4] <= pivot_score) --right;
    if (left <= right) {
      for (int i = 0; i < 5; ++i) {
        temp[i] = list_cpu[left * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        list_cpu[left * 5 + i] = list_cpu[right * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        list_cpu[right * 5 + i] = temp[i];
      }
      ++left;
      --right;
    }
  }

  if (right > start) {
    for (int i = 0; i < 5; ++i) {
      temp[i] = list_cpu[start * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      list_cpu[start * 5 + i] = list_cpu[right * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      list_cpu[right * 5 + i] = temp[i];
    }
  }

  if (start < right - 1) {
    sort_box(list_cpu, start, right - 1, num_top);
  }
  if (right + 1 < num_top && right + 1 < end) {
    sort_box(list_cpu, right + 1, end, num_top);
  }
}

template <typename Dtype>
__global__
static
void enumerate_proposals_gpu(const int nthreads,
                             const Dtype bottom4d[],
                             const Dtype d_anchor4d[],
                             const Dtype anchors[],
                             Dtype proposals[],
                             const int num_anchors,
                             const int bottom_H, const int bottom_W,
                             const Dtype img_H, const Dtype img_W,
                             const Dtype min_box_H, const Dtype min_box_W,
                             const int feat_stride)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int h = index / num_anchors / bottom_W;
    const int w = (index / num_anchors) % bottom_W;
    const int k = index % num_anchors;
    const Dtype x = w * feat_stride;
    const Dtype y = h * feat_stride;
    const Dtype* p_box = d_anchor4d + h * bottom_W + w;
    const Dtype* p_score = bottom4d + h * bottom_W + w;

    const int bottom_area = bottom_H * bottom_W;
    const Dtype dx = p_box[(k * 4 + 0) * bottom_area];
    const Dtype dy = p_box[(k * 4 + 1) * bottom_area];
    const Dtype d_log_w = p_box[(k * 4 + 2) * bottom_area];
    const Dtype d_log_h = p_box[(k * 4 + 3) * bottom_area];

    Dtype* const p_proposal = proposals + index * 5;
    p_proposal[0] = x + anchors[k * 4 + 0];
    p_proposal[1] = y + anchors[k * 4 + 1];
    p_proposal[2] = x + anchors[k * 4 + 2];
    p_proposal[3] = y + anchors[k * 4 + 3];
    p_proposal[4]
        = transform_box(p_proposal,
                        dx, dy, d_log_w, d_log_h,
                        img_W, img_H, min_box_W, min_box_H)
          * p_score[k * bottom_area];
  }
}

template <typename Dtype>
__global__
static
void retrieve_rois_gpu(const int nthreads,
                       const int item_index,
                       const Dtype proposals[],
                       const int roi_indices[],
                       Dtype rois[],
                       Dtype roi_scores[])
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    const Dtype* const proposals_index = proposals + roi_indices[index] * 5;
    rois[index * 5 + 0] = item_index;
    rois[index * 5 + 1] = proposals_index[0];
    rois[index * 5 + 2] = proposals_index[1];
    rois[index * 5 + 3] = proposals_index[2];
    rois[index * 5 + 4] = proposals_index[3];
    if (roi_scores) {
      roi_scores[index] = proposals_index[4];
    }
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top)
{
  CHECK_EQ(bottom[0]->shape(0), 1) << "Only single item batches are supported";

  const Dtype* p_bottom_item = bottom[0]->gpu_data();
  const Dtype* p_d_anchor_item = bottom[1]->gpu_data();
  const Dtype* p_img_info_cpu = bottom[2]->cpu_data();
  Dtype* p_roi_item = top[0]->mutable_gpu_data();
  Dtype* p_score_item = (top.size() > 1) ? top[1]->mutable_gpu_data() : NULL;

  vector<int> proposals_shape(2);
  vector<int> top_shape(2);
  proposals_shape[0] = 0;
  proposals_shape[1] = 5;
  top_shape[0] = 0;
  top_shape[1] = 5;

  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    // bottom shape: (2 x num_anchors) x H x W
    const int bottom_H = bottom[0]->height();
    const int bottom_W = bottom[0]->width();
    // input image height & width
    const Dtype img_H = p_img_info_cpu[0];
    const Dtype img_W = p_img_info_cpu[1];
    // scale factor for height & width
    const Dtype scale_H = p_img_info_cpu[2];
    const Dtype scale_W = p_img_info_cpu[3];
    // minimum box width & height
    const Dtype min_box_H = min_size_ * scale_H;
    const Dtype min_box_W = min_size_ * scale_W;
    // number of all proposals = num_anchors * H * W
    const int num_proposals = anchors_.shape(0) * bottom_H * bottom_W;
    // number of top-n proposals before NMS
    const int pre_nms_topn = std::min(num_proposals,  pre_nms_topn_);
    // number of final RoIs
    int num_rois = 0;

    // enumerate all proposals
    //   num_proposals = num_anchors * H * W
    //   (x1, y1, x2, y2, score) for each proposal
    // NOTE: for bottom, only foreground scores are passed
    proposals_shape[0] = num_proposals;
    proposals_.Reshape(proposals_shape);
    enumerate_proposals_gpu<Dtype><<<CAFFE_GET_BLOCKS(num_proposals),
                                     CAFFE_CUDA_NUM_THREADS>>>(
        num_proposals,
        p_bottom_item + num_proposals,  p_d_anchor_item,
        anchors_.gpu_data(),  proposals_.mutable_gpu_data(),  anchors_.shape(0),
        bottom_H,  bottom_W,  img_H,  img_W,  min_box_H,  min_box_W,
        feat_stride_);
    CUDA_POST_KERNEL_CHECK;

    sort_box(proposals_.mutable_cpu_data(), 0, num_proposals - 1, pre_nms_topn_);

    nms_gpu(pre_nms_topn,  proposals_.gpu_data(),  &nms_mask_,
            roi_indices_.mutable_cpu_data(),  &num_rois,
            0,  nms_thresh_,  post_nms_topn_);

    retrieve_rois_gpu<Dtype><<<CAFFE_GET_BLOCKS(num_rois),
                               CAFFE_CUDA_NUM_THREADS>>>(
        num_rois,  n,  proposals_.gpu_data(),  roi_indices_.gpu_data(),
        p_roi_item,  p_score_item);
    CUDA_POST_KERNEL_CHECK;

    top_shape[0] += num_rois;
  }

  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    top_shape.pop_back();
    top[1]->Reshape(top_shape);
  }
}
#endif

INSTANTIATE_LAYER_GPU_FUNCS(ProposalLayer);


}  // namespace caffe
