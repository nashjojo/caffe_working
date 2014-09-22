// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

// Kaixiang MO, 29th August, 2014
// sqrt element-wise
template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a,
    Dtype* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    y[index] = sqrtf(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a,
    float* y) {
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a,
    double* y) {
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

// inverse element-wise (x^-1)
template <typename Dtype>
__global__ void inverse_kernel(const int n, const Dtype* a,
    Dtype* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    y[index] = Dtype(1.) / (a[index]);
  }
}

template <>
void caffe_gpu_inverse<float>(const int N, const float* a,
    float* y) {
  inverse_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_inverse<double>(const int N, const double* a,
    double* y) {
  inverse_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}
// ~Kaixiang MO

}  // namespace caffe
