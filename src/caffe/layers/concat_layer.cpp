#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	concat_dim_ = this->layer_param_.concat_dim();
	CHECK_GE(concat_dim_, 0) <<
		"concat_dim should be >= 0";
	CHECK_LE(concat_dim_, 1) <<
		"For now concat_dim <=1, it can only concat num and channels";

	// call Reshape directly
	this->Reshape(bottom, top);
}

template <typename Dtype>
void ConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	// Initialize with the first blob.
	count_ = bottom[0]->count();
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	for (int i = 1; i < bottom.size(); ++i) {
		count_ += bottom[i]->count();
		if (concat_dim_== 0) {
			num_ += bottom[i]->num();
		} else if (concat_dim_ == 1) {
			channels_ += bottom[i]->channels();
		} else if (concat_dim_ == 2) {
			height_ += bottom[i]->height();
		} else if (concat_dim_ == 3) {
			width_ += bottom[i]->width();
		}
	}
	(*top)[0]->Reshape(num_, channels_, height_, width_);
	CHECK_EQ(count_, (*top)[0]->count());
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	if (concat_dim_== 0) {
		int offset_num = 0;
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			int num_elem = bottom[i]->count();
			// caffe_copy(num_elem, bottom_data, top_data+(*top)[0]->offset(offset_num));
			memcpy(top_data+(*top)[0]->offset(offset_num), bottom_data, sizeof(Dtype) * num_elem);

			offset_num += bottom[i]->num();
		}
	} else if (concat_dim_ == 1) {
		int offset_channel = 0;
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			int num_elem =
				bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
			for (int n = 0; n < num_; ++n) {
				// caffe_copy(num_elem, bottom_data+bottom[i]->offset(n),
				// 	top_data+(*top)[0]->offset(n, offset_channel));
				memcpy(top_data+(*top)[0]->offset(n, offset_channel), bottom_data+bottom[i]->offset(n), sizeof(Dtype) * num_elem);
			}
			offset_channel += bottom[i]->channels();
		}  // concat_dim_ is guaranteed to be 0 or 1 by LayerSetUp.
	}
}

template <typename Dtype>
Dtype ConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down) {
		const Dtype* top_diff = top[0]->cpu_diff();
		if (concat_dim_ == 0) {
			int offset_num = 0;
			for (int i = 0; i < bottom->size(); ++i) {
				Blob<Dtype>* blob = (*bottom)[i];
				Dtype* bottom_diff = blob->mutable_cpu_diff();
				// caffe_copy(blob->count(), top_diff + top[0]->offset(offset_num),
				// 					 bottom_diff);
				memcpy(bottom_diff, top_diff + top[0]->offset(offset_num), sizeof(Dtype) * blob->count());
				offset_num += blob->num();
			}
		} else if (concat_dim_ == 1) {
			int offset_channel = 0;
			for (int i = 0; i < bottom->size(); ++i) {
				Blob<Dtype>* blob = (*bottom)[i];
				Dtype* bottom_diff = blob->mutable_cpu_diff();
				int num_elem = blob->channels()*blob->height()*blob->width();
				for (int n = 0; n < num_; ++n) {
					// caffe_copy(num_elem, top_diff + top[0]->offset(n, offset_channel),
					// 					 bottom_diff + blob->offset(n));
					memcpy(bottom_diff + blob->offset(n), top_diff + top[0]->offset(n, offset_channel), sizeof(Dtype) * num_elem);
				}
				offset_channel += blob->channels();
			}
		}  // concat_dim_ is guaranteed to be 0 or 1 by LayerSetUp.
	} // propagate_down
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	Dtype* top_data = (*top)[0]->mutable_gpu_data();
	if (concat_dim_ == 0) {
		int offset_num = 0;
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			// caffe_copy(bottom[i]->count(), bottom_data,
			// 	top_data + (*top)[0]->offset(offset_num));
			CUDA_CHECK(cudaMemcpy(top_data + (*top)[0]->offset(offset_num), bottom_data, sizeof(Dtype) * bottom[i]->count(), cudaMemcpyDefault));
			offset_num += bottom[i]->num();
		}
	} else if (concat_dim_ == 1) {
		int offset_channel = 0;
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			int num_elem =
				bottom[i]->channels() * bottom[i]->height() * bottom[i]->width();
			for (int n = 0; n < num_; ++n) {
				// caffe_copy(num_elem, bottom_data+bottom[i]->offset(n),
				// 	top_data + (*top)[0]->offset(n, offset_channel));
				CUDA_CHECK(cudaMemcpy(top_data + (*top)[0]->offset(n, offset_channel), bottom_data+bottom[i]->offset(n), sizeof(Dtype) * num_elem, cudaMemcpyDefault));
			}
			offset_channel += bottom[i]->channels();
		}
	} else {
		LOG(FATAL) << "concat_dim along dim" << concat_dim_ <<
			" not implemented yet";
	}
}

template <typename Dtype>
Dtype ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down) {
		const Dtype* top_diff = top[0]->gpu_diff();
		if (concat_dim_ == 0) {
			int offset_num = 0;
			for (int i = 0; i < bottom->size(); ++i) {
				Blob<Dtype>* blob = (*bottom)[i];
				Dtype* bottom_diff = blob->mutable_gpu_diff();
				// caffe_copy(blob->count(), top_diff + top[0]->offset(offset_num),
				// 							 bottom_diff);
				CUDA_CHECK(cudaMemcpy(bottom_diff, top_diff + top[0]->offset(offset_num), sizeof(Dtype) * blob->count(), cudaMemcpyDefault));
				offset_num += blob->num();
			}
		} else if (concat_dim_ == 1) {
			int offset_channel = 0;
			for (int i = 0; i < bottom->size(); ++i) {
				Blob<Dtype>* blob = (*bottom)[i];
				Dtype* bottom_diff = blob->mutable_gpu_diff();
				int num_elem = blob->channels()*blob->height()*blob->width();
				for (int n = 0; n < num_; ++n) {
					// caffe_copy(num_elem, top_diff + top[0]->offset(n, offset_channel),
					// 							 bottom_diff + blob->offset(n));
					CUDA_CHECK(cudaMemcpy(bottom_diff + blob->offset(n), top_diff + top[0]->offset(n, offset_channel), sizeof(Dtype) * num_elem, cudaMemcpyDefault));
				}
				offset_channel += blob->channels();
			}
		} else {
			LOG(FATAL) << "concat_dim along dim" << concat_dim_ <<
				" not implemented yet";
		}
	}  // propagate_down
}

INSTANTIATE_CLASS(ConcatLayer);

}   // namespace caffe