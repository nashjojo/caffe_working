// Copyright 2014 Tobias Domhan

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include <algorithm>
#include <cmath>

// Kaixiang Mo
#include <iostream>
// ~Kaixiang Mo

using std::max;

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}

template <typename Dtype>
Dtype SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    for (int i = 0; i < count; ++i) {
      Dtype sigmoid_x = sigmoid(bottom_data[i]);
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
  return Dtype(0);
}


template <typename Dtype>
__device__ inline Dtype sigmoid_gpu(Dtype x) {
  return 1. / (1. + exp(-x));
}


template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out[index] = sigmoid_gpu(in[index]);
  }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
	
  // int num = bottom[0]->num();
  // int dim = bottom[0]->count() / num;
  // const Dtype* bottom_data1 = bottom[0]->cpu_data();
  // std::cout << "SigmoidLayer input" << std::endl;
  // for (int i = 0; i < 5; ++i) {
	// for (int j = 0; j < std::min(10,dim); ++j ){
	  // std::cout << bottom_data1[i * dim + j] << "\t";
	// }
	// std::cout << std::endl; 
  // }

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  
  SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
  
  // const Dtype* top_data1 = (*top)[0]->cpu_data();
  // std::cout << "SigmoidLayer output" << std::endl;
  // for (int i = 0; i < 5; ++i) {
	// for (int j = 0; j < std::min(10,dim); ++j ){
	  // std::cout << top_data1[i * dim + j] << "\t";
	// }
	// std::cout << std::endl; 
  // }
  
}

template <typename Dtype>
__global__ void SigmoidBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    Dtype sigmoid_x = sigmoid_gpu(in_data[index]);
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}

template <typename Dtype>
Dtype SigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    SigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(SigmoidLayer);


}  // namespace caffe
