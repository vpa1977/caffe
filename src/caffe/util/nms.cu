#include "caffe/util/nms.hpp"

#define DIV_THEN_CEIL(x, y)  (((x) + (y) - 1) / (y))

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#ifdef USE_GREENTEA
static const int nms_block_size = 64;
const char* const nms_kernel = 
		"\n"
		"#define nms_block_size 64\n"
		"\n"
		"\n"
		"Dtype iou(__global const Dtype* A, __local const Dtype* B)\n"
		"{\n"
		"	\n"
		"  // overlapped region (= box)\n"
		"  const Dtype x1 = max(A[0],  B[0]);\n"
"  const Dtype y1 = max(A[1],  B[1]);\n"
"  const Dtype x2 = min(A[2],  B[2]);\n"
"  const Dtype y2 = min(A[3],  B[3]);\n"
"\n"
"  // intersection area\n"
"  const Dtype width = max((Dtype)0,  x2 - x1 + (Dtype)1);\n"
"  const Dtype height = max((Dtype)0,  y2 - y1 + (Dtype)1);\n"
"  const Dtype area = width * height;\n"
"\n"
"  // area of A, B\n"
"  const Dtype A_area = (A[2] - A[0] + (Dtype)1) * (A[3] - A[1] + (Dtype)1);\n"
"  const Dtype B_area = (B[2] - B[0] + (Dtype)1) * (B[3] - B[1] + (Dtype)1);\n"
"\n"
"  // IoU\n"
"  return area / (A_area + B_area - area);\n"
"	\n"
"	return 0;\n"
"}\n"
"\n"
"\n"
"\n"
"// given box proposals, compute overlap between all box pairs\n"
"// (overlap = intersection area / union area)\n"
"// and then set mask-bit to 1 if a pair is significantly overlapped\n"
"//   num_boxes: number of box proposals given\n"
"//   boxes: 'num_boxes x 5' array (x1, y1, x2, y2, score)\n"
"//   nms_thresh: threshold for determining 'significant overlap'\n"
"//               if 'intersection area / union area > nms_thresh',\n"
"//               two boxes are thought of as significantly overlapped\n"
"// the all-pair computation (num_boxes x num_boxes) is done by\n"
"// divide-and-conquer:\n"
"//   each GPU block (bj, bi) computes for '64 x 64' box pairs (j, i),\n"
"//     j = bj * 64 + { 0, 1, ..., 63 }\n"
"//     i = bi * 64 + { 0, 1, ..., 63 },\n"
"//   and each '1 x 64' results is saved into a 64-bit mask\n"
"//     mask: 'num_boxes x num_blocks' array\n"
"//     for mask[j][bi], 'di-th bit = 1' means:\n"
"//       box j is significantly overlapped with box i,\n"
"//       where i = bi * 64 + di\n"
"__kernel void nms_mask(__global const Dtype* boxes, __global unsigned long long* mask,\n"
"              const int num_boxes, const Dtype nms_thresh)\n"
"{\n"
"	__local Dtype boxes_i[nms_block_size * 4];\n"
"	const int di = get_local_id(0);\n"
"	const int dj = get_local_id(0);\n"
"	const int bi = get_group_id(0);	\n"
"	const int num_blocks = (num_boxes+  nms_block_size -1)/nms_block_size;\n"
"  // block region\n"
"  //   j = j_start + { 0, ..., dj_end - 1 }\n"
"  //   i = i_start + { 0, ..., di_end - 1 }\n"
"  const int i_start = get_group_id(0) * nms_block_size;\n"
"  const int di_end = min(num_boxes - i_start,  nms_block_size);\n"
"	\n"
"  const int j_start = get_group_id(1) * nms_block_size;\n"
"  const int dj_end = min(num_boxes - j_start,  nms_block_size);\n"
"\n"
"  // copy all i-th boxes to GPU cache\n"
"  //   i = i_start + { 0, ..., di_end - 1 }\n"
"	if (di < di_end) {\n"
"		boxes_i[di * 4 + 0] = boxes[(i_start + di) * 5 + 0];\n"
"		boxes_i[di * 4 + 1] = boxes[(i_start + di) * 5 + 1];\n"
"		boxes_i[di * 4 + 2] = boxes[(i_start + di) * 5 + 2];\n"
"		boxes_i[di * 4 + 3] = boxes[(i_start + di) * 5 + 3];\n"
"	}\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"  // given j = j_start + dj,\n"
"  //   check whether box i is significantly overlapped with box j\n"
"  //   (i.e., IoU(box j, box i) > threshold)\n"
"  //   for all i = i_start + { 0, ..., di_end - 1 } except for i == j\n"
"	\n"
"	if (dj < dj_end) {\n"
"		// box j\n"
"		__global const Dtype* const box_j = boxes + (j_start + dj) * 5;\n"
"\n"
"		// mask for significant overlap\n"
"		//   if IoU(box j, box i) > threshold,  di-th bit = 1\n"
"		unsigned long long mask_j = 0;\n"
"\n"
"		// check for all i = i_start + { 0, ..., di_end - 1 }\n"
"		// except for i == j\n"
"		const int di_start = (i_start == j_start) ? (dj + 1) : 0;\n"
"		for (int di = di_start; di < di_end; ++di) {\n"
"			// box i\n"
"			__local const Dtype* const box_i = boxes_i + di * 4;\n"
"\n"
"			// if IoU(box j, box i) > threshold,  di-th bit = 1\n"
"			if (iou(box_j, box_i) > nms_thresh) {\n"
"				mask_j |= 1ULL << di;\n"
"			}\n"
"		}\n"
"		unsigned long long val = mask_j;\n"
"		// mask: 'num_boxes x num_blocks' array\n"
"		//   for mask[j][bi], 'di-th bit = 1' means:\n"
"		//     box j is significantly overlapped with box i = i_start + di,\n"
"		//     where i_start = bi * block_size\n"
"		const int offset = (j_start + dj) * num_blocks + bi;\n"
"		mask[offset] = val;\n"
"	} // endif dj < dj_end\n"
"}\n"
"\n";


// given box proposals (sorted in descending order of their scores),
// discard a box if it is significantly overlapped with
// one or more previous (= scored higher) boxes
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//          sorted in descending order of scores
//   aux_data: auxiliary data for NMS operation
//   num_out: number of remaining boxes
//   index_out_cpu: "num_out x 1" array
//                  indices of remaining boxes
//                  allocated at main memory
//   base_index: a constant added to index_out_cpu,  usually 0
//               index_out_cpu[i] = base_index + actual index in boxes
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
//   bbox_vote: whether bounding-box voting is used (= 1) or not (= 0)
//   vote_thresh: threshold for selecting overlapped boxes
//                which are participated in bounding-box voting
template <typename Dtype>
void nms_gpu(
	const int device_id,
	const int num_boxes,
	const Dtype boxes_gpu[],
	Blob<int>* const p_mask,
	int index_out_cpu[],
	int* const num_out,
	const int base_index,
	const Dtype nms_thresh, const int max_num_out)
{
	const int num_blocks = DIV_THEN_CEIL(num_boxes, nms_block_size);

	{
		vector<int> mask_shape(2);
		mask_shape[0] = num_boxes;
		mask_shape[1] = num_blocks * sizeof(unsigned long long) / sizeof(int);
		p_mask->Reshape(mask_shape);

		viennacl::ocl::context &ctx = viennacl::ocl::get_context(device_id);
		static bool compiled = false;
		if (!compiled)
		{
			std::string kernel;
			if (is_same<Dtype, float>::value)
				kernel = "#define Dtype float\n";
			else if (is_same<Dtype, double>::value)
				kernel = "#define Dtype double\n";
			kernel += nms_kernel;
			ctx.add_program(kernel.c_str(), CL_KERNEL_SELECT("nms"));
			compiled = true;
		}
		viennacl::ocl::program &program = ctx.get_program(CL_KERNEL_SELECT("nms"));
		viennacl::ocl::kernel &nms = program.get_kernel("nms_mask");
		nms.global_work_size(0, num_blocks * nms_block_size);
		nms.local_work_size(0, nms_block_size);
		viennacl::ocl::enqueue(
			nms(WrapHandle((cl_mem)boxes_gpu, &ctx),
				WrapHandle((cl_mem)p_mask->mutable_gpu_data(), &ctx),
				num_boxes, nms_thresh),
			ctx.get_queue());
	}

	// discard i-th box if it is significantly overlapped with
	// one or more previous (= scored higher) boxes
	{
		const unsigned long long* const p_mask_cpu
			= (unsigned long long*)p_mask->cpu_data();
		int num_selected = 0;
		vector<unsigned long long> dead_bit(num_blocks);
		for (int i = 0; i < num_blocks; ++i) {
			dead_bit[i] = 0;
		}

		for (int i = 0; i < num_boxes; ++i) {
			const int nblock = i / nms_block_size;
			const int inblock = i % nms_block_size;

			if (!(dead_bit[nblock] & (1ULL << inblock))) {
				index_out_cpu[num_selected++] = base_index + i;
				const unsigned long long* const mask_i = p_mask_cpu + i * num_blocks;
				for (int j = nblock; j < num_blocks; ++j) {
					dead_bit[j] |= mask_i[j];
				}

				if (num_selected == max_num_out) {
					break;
				}
			}
		}
		*num_out = num_selected;
	}
}

template
void nms_gpu(const int device_id,const int num_boxes,
	const float boxes_gpu[],
	Blob<int>* const p_mask,
	int index_out_cpu[],
	int* const num_out,
	const int base_index,
	const float nms_thresh, const int max_num_out);
template
void nms_gpu(const int device_id, const int num_boxes,
	const double boxes_gpu[],
	Blob<int>* const p_mask,
	int index_out_cpu[],
	int* const num_out,
	const int base_index,
	const double nms_thresh, const int max_num_out);

#endif

#ifdef USE_CUDA
template <typename Dtype>
__device__
static
Dtype iou(const Dtype A[], const Dtype B[])
{
  // overlapped region (= box)
  const Dtype x1 = max(A[0],  B[0]);
  const Dtype y1 = max(A[1],  B[1]);
  const Dtype x2 = min(A[2],  B[2]);
  const Dtype y2 = min(A[3],  B[3]);

  // intersection area
  const Dtype width = max((Dtype)0,  x2 - x1 + (Dtype)1);
  const Dtype height = max((Dtype)0,  y2 - y1 + (Dtype)1);
  const Dtype area = width * height;

  // area of A, B
  const Dtype A_area = (A[2] - A[0] + (Dtype)1) * (A[3] - A[1] + (Dtype)1);
  const Dtype B_area = (B[2] - B[0] + (Dtype)1) * (B[3] - B[1] + (Dtype)1);

  // IoU
  return area / (A_area + B_area - area);
}

static const int nms_block_size = 64;

// given box proposals, compute overlap between all box pairs
// (overlap = intersection area / union area)
// and then set mask-bit to 1 if a pair is significantly overlapped
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
// the all-pair computation (num_boxes x num_boxes) is done by
// divide-and-conquer:
//   each GPU block (bj, bi) computes for "64 x 64" box pairs (j, i),
//     j = bj * 64 + { 0, 1, ..., 63 }
//     i = bi * 64 + { 0, 1, ..., 63 },
//   and each "1 x 64" results is saved into a 64-bit mask
//     mask: "num_boxes x num_blocks" array
//     for mask[j][bi], "di-th bit = 1" means:
//       box j is significantly overlapped with box i,
//       where i = bi * 64 + di
template <typename Dtype>
__global__
static
void nms_mask(const Dtype boxes[], unsigned long long mask[],
              const int num_boxes, const Dtype nms_thresh)
{
  // block region
  //   j = j_start + { 0, ..., dj_end - 1 }
  //   i = i_start + { 0, ..., di_end - 1 }
  const int i_start = blockIdx.x * nms_block_size;
  const int di_end = min(num_boxes - i_start,  nms_block_size);
  const int j_start = blockIdx.y * nms_block_size;
  const int dj_end = min(num_boxes - j_start,  nms_block_size);

  // copy all i-th boxes to GPU cache
  //   i = i_start + { 0, ..., di_end - 1 }
  __shared__ Dtype boxes_i[nms_block_size * 4];
  {
    const int di = threadIdx.x;
    if (di < di_end) {
      boxes_i[di * 4 + 0] = boxes[(i_start + di) * 5 + 0];
      boxes_i[di * 4 + 1] = boxes[(i_start + di) * 5 + 1];
      boxes_i[di * 4 + 2] = boxes[(i_start + di) * 5 + 2];
      boxes_i[di * 4 + 3] = boxes[(i_start + di) * 5 + 3];
    }
  }
  __syncthreads();

  // given j = j_start + dj,
  //   check whether box i is significantly overlapped with box j
  //   (i.e., IoU(box j, box i) > threshold)
  //   for all i = i_start + { 0, ..., di_end - 1 } except for i == j
  {
    const int dj = threadIdx.x;
    if (dj < dj_end) {
      // box j
      const Dtype* const box_j = boxes + (j_start + dj) * 5;

      // mask for significant overlap
      //   if IoU(box j, box i) > threshold,  di-th bit = 1
      unsigned long long mask_j = 0;

      // check for all i = i_start + { 0, ..., di_end - 1 }
      // except for i == j
      const int di_start = (i_start == j_start) ? (dj + 1) : 0;
      for (int di = di_start; di < di_end; ++di) {
        // box i
        const Dtype* const box_i = boxes_i + di * 4;

        // if IoU(box j, box i) > threshold,  di-th bit = 1
        if (iou(box_j, box_i) > nms_thresh) {
          mask_j |= 1ULL << di;
        }
      }

      // mask: "num_boxes x num_blocks" array
      //   for mask[j][bi], "di-th bit = 1" means:
      //     box j is significantly overlapped with box i = i_start + di,
      //     where i_start = bi * block_size
      {
        const int num_blocks = DIV_THEN_CEIL(num_boxes,  nms_block_size);
        const int bi = blockIdx.x;
        mask[(j_start + dj) * num_blocks + bi] = mask_j;
      }
    } // endif dj < dj_end
  }
}

// given box proposals (sorted in descending order of their scores),
// discard a box if it is significantly overlapped with
// one or more previous (= scored higher) boxes
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//          sorted in descending order of scores
//   aux_data: auxiliary data for NMS operation
//   num_out: number of remaining boxes
//   index_out_cpu: "num_out x 1" array
//                  indices of remaining boxes
//                  allocated at main memory
//   base_index: a constant added to index_out_cpu,  usually 0
//               index_out_cpu[i] = base_index + actual index in boxes
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
//   bbox_vote: whether bounding-box voting is used (= 1) or not (= 0)
//   vote_thresh: threshold for selecting overlapped boxes
//                which are participated in bounding-box voting
template <typename Dtype>
void nms_gpu(const int num_boxes,
             const Dtype boxes_gpu[],
             Blob<int>* const p_mask,
             int index_out_cpu[],
             int* const num_out,
             const int base_index,
             const Dtype nms_thresh, const int max_num_out)
{
  const int num_blocks = DIV_THEN_CEIL(num_boxes,  nms_block_size);

  {
    const dim3 blocks(num_blocks, num_blocks);
    vector<int> mask_shape(2);
    mask_shape[0] = num_boxes;
    mask_shape[1] = num_blocks * sizeof(unsigned long long) / sizeof(int);
    p_mask->Reshape(mask_shape);

    // find all significantly-overlapped pairs of boxes
    nms_mask<<<blocks, nms_block_size>>>(
        boxes_gpu,  (unsigned long long*)p_mask->mutable_gpu_data(),
        num_boxes,  nms_thresh);
    CUDA_POST_KERNEL_CHECK;
  }

  // discard i-th box if it is significantly overlapped with
  // one or more previous (= scored higher) boxes
  {
    const unsigned long long* const p_mask_cpu
         = (unsigned long long*)p_mask->cpu_data();
    int num_selected = 0;
    vector<unsigned long long> dead_bit(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
      dead_bit[i] = 0;
    }

    for (int i = 0; i < num_boxes; ++i) {
      const int nblock = i / nms_block_size;
      const int inblock = i % nms_block_size;

      if (!(dead_bit[nblock] & (1ULL << inblock))) {
        index_out_cpu[num_selected++] = base_index + i;
        const unsigned long long* const mask_i = p_mask_cpu + i * num_blocks;
        for (int j = nblock; j < num_blocks; ++j) {
          dead_bit[j] |= mask_i[j];
        }

        if (num_selected == max_num_out) {
          break;
        }
      }
    }
    *num_out = num_selected;
  }
}

template
void nms_gpu(const int num_boxes,
	const float boxes_gpu[],
	Blob<int>* const p_mask,
	int index_out_cpu[],
	int* const num_out,
	const int base_index,
	const float nms_thresh, const int max_num_out);
template
void nms_gpu(const int num_boxes,
	const double boxes_gpu[],
	Blob<int>* const p_mask,
	int index_out_cpu[],
	int* const num_out,
	const int base_index,
	const double nms_thresh, const int max_num_out);

#endif


}  // namespace caffe
