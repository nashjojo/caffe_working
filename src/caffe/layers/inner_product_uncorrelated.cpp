// Copyright 2013 Yangqing Jia


#include <mkl.h>
#include <cublas_v2.h>

#include <vector>

// Kaixiang MO
#include <iostream>
#include <iomanip>
// ~Kaixiang Mo

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayerUncorrelated<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "IP Layer takes 2 blobs as input, 2nd is the extra features.";
	CHECK_EQ(top->size(), 1) << "IP Layer takes a single blob as output.";
	const int num_output = this->layer_param_.num_output();
	num_extfeature = bottom[1]->channels(); // is bottom[1] initialized yet?
	CHECK(num_extfeature > 0) << "Number of extra feature should be larger than 0.";

	biasterm_ = this->layer_param_.biasterm();
	// Figure out the dimensions
	M_ = bottom[0]->num();
	K_ = bottom[0]->count() / bottom[0]->num();
	N_ = num_output;
	(*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);
	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		if (biasterm_) {
			this->blobs_.resize(2);
		} else {
			this->blobs_.resize(1);
		}
		// Intialize the weight
		this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
		// fill the weights
		shared_ptr<Filler<Dtype> > weight_filler(
				GetFiller<Dtype>(this->layer_param_.weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
		// If necessary, intiialize and fill the bias term
		if (biasterm_) {
			this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));
			shared_ptr<Filler<Dtype> > bias_filler(
					GetFiller<Dtype>(this->layer_param_.bias_filler()));
			bias_filler->Fill(this->blobs_[1].get());
		}
	}  // parameter initialization
	// Setting up the bias multiplier
	if (biasterm_) {
		bias_multiplier_.reset(new SyncedMemory(M_ * sizeof(Dtype)));
		Dtype* bias_multiplier_data =
				reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
		for (int i = 0; i < M_; ++i) {
				bias_multiplier_data[i] = 1.;
		}
	}

	// Setting up the blobs for mean and feature-mean, and final correlation matrix
	feature_mean_blobs.resize(5);
	feature_mean_blobs[0].reset(new Blob<Dtype>(1, 1, 1, N_));
	feature_mean_blobs[1].reset(new Blob<Dtype>(1, 1, 1, num_extfeature));
	feature_mean_blobs[2].reset(new Blob<Dtype>(M_, N_, 1, 1));
	feature_mean_blobs[3].reset(new Blob<Dtype>(M_, num_extfeature, 1, 1));
	feature_mean_blobs[4].reset(new Blob<Dtype>(1, 1, num_extfeature, N_));
};

template <typename Dtype>
void InnerProductLayerUncorrelated<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, weight, (Dtype)0., top_data);
	if (biasterm_) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
				this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
	}
}

template <typename Dtype>
Dtype InnerProductLayerUncorrelated<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {

	Dtype lambda = this->layer_param_.covar_factor();
	// const Dtype* top_diff = top[0]->cpu_diff(); // need to change the value later.
	// top_diff have the same shape with top_data, M_*N_
	const Dtype* top_data = top[0]->cpu_data();
	// const Dtype* bottom_data = (*bottom)[0]->cpu_data();
	const Dtype* extfeature_data = (*bottom)[1]->cpu_data();

	/*
	0: output feature means
	1: extfeature means 
	2: output feature - mean
	3: extfeature - mean
	4: covariance matrix
	*/
	// vector<shared_ptr<Blob<Dtype> > > feature_mean_blobs;
	// feature_mean_blobs.resize(5);
	// feature_mean_blobs[0].reset(new Blob<Dtype>(1, 1, 1, N_));
	// feature_mean_blobs[1].reset(new Blob<Dtype>(1, 1, 1, num_extfeature));
	// feature_mean_blobs[2].reset(new Blob<Dtype>(M_, N_, 1, 1));
	// feature_mean_blobs[3].reset(new Blob<Dtype>(M_, num_extfeature, 1, 1));
	// feature_mean_blobs[4].reset(new Blob<Dtype>(1, 1, num_extfeature, N_));

	// mean for output: after careful consideration
	// 
	caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1. / M_, top_data,
			reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)0., 
			feature_mean_blobs[0]->mutable_cpu_data());

	// mean for extfeature
	caffe_cpu_gemv<Dtype>(CblasTrans, M_, num_extfeature, (Dtype)1. / M_, extfeature_data,
			reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)0., 
			feature_mean_blobs[1]->mutable_cpu_data());

	// output feature - mean
	memcpy(feature_mean_blobs[2]->mutable_cpu_data(),
			top_data, sizeof(Dtype) * top[0]->count());
	// Dtype* top_data_no_mean = feature_mean_blobs[2]->mutable_cpu_data();
	const Dtype* top_data_mean = feature_mean_blobs[0]->cpu_data();
	
	// for (int i = 0; i < N_; ++i) {
	// 	for (int j = 0; j < M_; ++j) {
	// 		top_data_no_mean[j*N_ + i] = top_data_no_mean[j*N_ + i] - top_data_mean[i];
	// 	}
	// }
	// use algebra operation to do it!
	for (int j = 0; j < M_; ++j) {
		caffe_axpy<Dtype>(N_, (Dtype)-1., top_data_mean, feature_mean_blobs[2]->mutable_cpu_data() + j*N_ );
	}

	// extfeature - mean
	memcpy(feature_mean_blobs[3]->mutable_cpu_data(),
			extfeature_data, sizeof(Dtype) * (*bottom)[1]->count());
	Dtype* extfeature_data_no_mean = feature_mean_blobs[3]->mutable_cpu_data();
	const Dtype* extfeature_data_mean = feature_mean_blobs[1]->cpu_data();
	
	// for (int i = 0; i < num_extfeature; ++i) {
	// 	for (int j = 0; j < M_; ++j) {
	// 		extfeature_data_no_mean[j*num_extfeature + i] = extfeature_data_no_mean[j*num_extfeature + i] - extfeature_data_mean[i];
	// 	}
	// }
	// use algebra operation to do it!
	for (int j = 0; j < M_; ++j) {
		caffe_axpy<Dtype>(num_extfeature, (Dtype)-1., extfeature_data_mean, feature_mean_blobs[3]->mutable_cpu_data() + j*num_extfeature);
	}

	// calculate correlation, 
	// (output-mean)*(feature-mean) in a single matrix multiplication
	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_extfeature, N_, M_, (Dtype)1.,
			feature_mean_blobs[3]->cpu_data(), feature_mean_blobs[2]->cpu_data(), (Dtype)0., feature_mean_blobs[4]->mutable_cpu_data());

	// modify top_diff so as to penalize large correlation
	// specific regularization factor
	// Dtype lambda = (Dtype)0.01;
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, num_extfeature, lambda / (num_extfeature* N_),
			feature_mean_blobs[3]->cpu_data(), feature_mean_blobs[4]->cpu_data(), (Dtype)1., top[0]->mutable_cpu_diff());

	// because we have modified top_diff
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = (*bottom)[0]->cpu_data();
	// Gradient with respect to weight
	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
	if (biasterm_) {
		// Gradient with respect to bias
		caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
				reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)0.,
				this->blobs_[1]->mutable_cpu_diff());
	}
	if (propagate_down) {
		// Gradient with respect to bottom data
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
				(*bottom)[0]->mutable_cpu_diff());
	}
	return Dtype(0);
}

template <typename Dtype>
void InnerProductLayerUncorrelated<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {

	// sample 10 bottom data and print out
	// const Dtype* bottom_data1 = bottom[0]->cpu_data();
	// std::cout << this->layer_param_.name() << " input" << std::endl;
	// for (int i = 0; i < 5; i++) {
	// std::cout << std::setw(10) << bottom_data1[i] <<"\t";
	// }
	// std::cout << std::endl;

	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = (*top)[0]->mutable_gpu_data();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, weight, (Dtype)0., top_data);
	if (biasterm_) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
				this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
	}
	
	// sample 10 bottom data and print out
	// const Dtype* top_data1 = (*top)[0]->cpu_data();
	// std::cout << this->layer_param_.name() << " output" << std::endl;
	// for (int i = 0; i < 5; i++) {
	// std::cout << std::setw(10) << top_data1[i] <<"\t";
	// }
	// std::cout << std::endl;
}

template <typename Dtype>
Dtype InnerProductLayerUncorrelated<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {

	Dtype lambda = this->layer_param_.covar_factor();
	std::cout<< "lambda=" << lambda << std::endl; 

	// const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* top_data = top[0]->gpu_data();
	// const Dtype* bottom_data = (*bottom)[0]->gpu_data();
	const Dtype* extfeature_data = (*bottom)[1]->gpu_data();

	// modify top_diff so as to penalize large correlation
	/*
	0: output feature means
	1: extfeature means 
	2: output feature - mean
	3: extfeature - mean
	4: covariance matrix
	*/
	// vector<shared_ptr<Blob<Dtype> > > feature_mean_blobs;
	// feature_mean_blobs.resize(5);
	// feature_mean_blobs[0].reset(new Blob<Dtype>(1, 1, 1, N_));
	// feature_mean_blobs[1].reset(new Blob<Dtype>(1, 1, 1, num_extfeature));
	// feature_mean_blobs[2].reset(new Blob<Dtype>(M_, N_, 1, 1));
	// feature_mean_blobs[3].reset(new Blob<Dtype>(M_, num_extfeature, 1, 1));
	// feature_mean_blobs[4].reset(new Blob<Dtype>(1, 1, num_extfeature, N_));

	// original data
	int num_col = 10; // number of columns to show
	// feature 1-5 for all instance
	// const Dtype* top_data_cpu = top[0]->cpu_data();
	// std::cout << "top_data" << std::endl;
	// for (int i = 0; i < M_; ++i){
	// 	for (int j = 0; j < min(N_, num_col); ++j) {
	// 		std::cout << top_data_cpu[i*N_+j] << ",\t";
	// 	}
	// 	std::cout << std::endl;
	// }
	// const Dtype* extfeature_data_cpu = (*bottom)[1]->cpu_data();
	// std::cout << "extfeature_data" << std::endl;
	// for (int i = 0; i < M_; ++i){
	// 	for (int j = 0; j < min(num_extfeature, num_col); ++j) {
	// 		std::cout << extfeature_data_cpu[i*num_extfeature+j] << ",\t";
	// 	}
	// 	std::cout << std::endl;
	// }

	// mean for output: after careful consideration
	caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1. / M_, top_data,
			reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), (Dtype)0., 
			feature_mean_blobs[0]->mutable_gpu_data());

	// mean for extfeature
	caffe_gpu_gemv<Dtype>(CblasTrans, M_, num_extfeature, (Dtype)1. / M_, extfeature_data,
			reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), (Dtype)0., 
			feature_mean_blobs[1]->mutable_gpu_data());

	// mean
	// feature 1-5 for all instance
	// const Dtype* top_data_mean_cpu = feature_mean_blobs[0]->cpu_data();
	// std::cout << "top_data mean" << std::endl;
	// for (int j = 0; j < min(N_, num_col); ++j) {
	// 	std::cout << top_data_mean_cpu[j] << ",\t";
	// }
	// std::cout << std::endl;
	// const Dtype* extfeature_data_mean_cpu = feature_mean_blobs[1]->cpu_data();
	// std::cout << "extfeature_data mean" << std::endl;
	// for (int j = 0; j < min(num_extfeature, num_col); ++j) {
	// 	std::cout << extfeature_data_mean_cpu[j] << ",\t";
	// }
	// std::cout << std::endl;

	// output feature - mean
	CUDA_CHECK(cudaMemcpy(feature_mean_blobs[2]->mutable_gpu_data(),
			top_data, sizeof(Dtype) * top[0]->count(), cudaMemcpyDeviceToDevice));
	// Dtype* top_data_no_mean = feature_mean_blobs[2]->mutable_gpu_data();
	const Dtype* top_data_mean = feature_mean_blobs[0]->gpu_data();
	
	// for (int i = 0; i < N_; ++i) {
	// 	for (int j = 0; j < M_; ++j) {
	// 		top_data_no_mean[j*N_ + i] = top_data_no_mean[j*N_ + i] - top_data_mean[i];
	// 	}
	// }
	// use algebra operation to do it!
	for (int j = 0; j < M_; ++j) {
		caffe_gpu_axpy<Dtype>(N_, (Dtype)-1., top_data_mean, feature_mean_blobs[2]->mutable_gpu_data() + j*N_ );
	}

	// extfeature - mean
	CUDA_CHECK(cudaMemcpy(feature_mean_blobs[3]->mutable_gpu_data(),
			extfeature_data, sizeof(Dtype) * (*bottom)[1]->count(), cudaMemcpyDeviceToDevice));
	// Dtype* extfeature_data_no_mean = feature_mean_blobs[3]->mutable_gpu_data();
	const Dtype* extfeature_data_mean = feature_mean_blobs[1]->gpu_data();
	
	// for (int i = 0; i < num_extfeature; ++i) {
	// 	for (int j = 0; j < M_; ++j) {
	// 		extfeature_data_no_mean[j*num_extfeature + i] = extfeature_data_no_mean[j*num_extfeature + i] - extfeature_data_mean[i];
	// 	}
	// }
	// use algebra operation to do it!
	for (int j = 0; j < M_; ++j) {
		caffe_gpu_axpy<Dtype>(num_extfeature, (Dtype)-1., extfeature_data_mean, feature_mean_blobs[3]->mutable_gpu_data() + j*num_extfeature);
	}

	// feature-mean
	// const Dtype* top_data_no_mean_cpu = feature_mean_blobs[2]->cpu_data();
	// std::cout << "top_data - mean" << std::endl;
	// for (int i = 0; i < M_; ++i){
	// 	for (int j = 0; j < min(N_, num_col); ++j) {
	// 		std::cout << top_data_no_mean_cpu[i*N_+j] << ",\t";
	// 	}
	// 	std::cout << std::endl;
	// }
	// const Dtype* extfeature_data_no_mean_cpu = feature_mean_blobs[3]->cpu_data();
	// std::cout << "extfeature_data - mean" << std::endl;
	// for (int i = 0; i < M_; ++i){
	// 	for (int j = 0; j < min(num_extfeature, num_col); ++j) {
	// 		std::cout << extfeature_data_no_mean_cpu[i*num_extfeature+j] << ",\t";
	// 	}
	// 	std::cout << std::endl;
	// }

	// calculate correlation, 
	// (output-mean)*(feature-mean) in a single matrix multiplication
	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_extfeature, N_, M_, (Dtype)1.,
			feature_mean_blobs[3]->gpu_data(), feature_mean_blobs[2]->gpu_data(), (Dtype)0., feature_mean_blobs[4]->mutable_gpu_data());

	// covariance matrix
	// const Dtype* covariance = feature_mean_blobs[4]->cpu_data();
	// std::cout << "covariance matrix" << std::endl;
	// for (int i = 0; i < num_extfeature; ++i){
	// 	for (int j = 0; j < min(N_, num_col); ++j) {
	// 		std::cout << covariance[i*N_+j] << ",\t";
	// 	}
	// 	std::cout << std::endl;
	// }


	// modify top_diff so as to penalize large correlation
	// specific regularization factor
	// Dtype lambda = (Dtype)0.01;
	// should normalize for more output features and more extra features
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, num_extfeature, lambda / (num_extfeature* N_),
			feature_mean_blobs[3]->gpu_data(), feature_mean_blobs[4]->gpu_data(), (Dtype)1., top[0]->mutable_gpu_diff());
	
	// changed (Dtype)1. to 0 for matrix multiplication result check
	// std::cout << "extfeature * covariance matrix" << std::endl;
	// for (int i = 0; i < M_; ++i){
	// 	for (int j = 0; j < min(N_, num_col); ++j) {
	// 		std::cout << top[0]->cpu_diff()[i*N_+j] << ",\t";
	// 	}
	// 	std::cout << std::endl;
	// }

	// we have modified top_diff
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = (*bottom)[0]->gpu_data();
	// Gradient with respect to weight
	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
	if (biasterm_) {
		// Gradient with respect to bias
		caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
				reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
				(Dtype)0., this->blobs_[1]->mutable_gpu_diff());
	}
	if (propagate_down) {
		// Gradient with respect to bottom data
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
				(*bottom)[0]->mutable_gpu_diff());
	}

	// std::cout << "weight matrix" << std::endl;
	// for (int i = 0; i < M_; ++i){
	// 	for (int j = 0; j < min(N_, num_col); ++j) {
	// 		std::cout << this->blobs_[0]->cpu_data()[i*N_+j] << ",\t";
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << "weight_diff matrix" << std::endl;
	// for (int i = 0; i < M_; ++i){
	// 	for (int j = 0; j < min(N_, num_col); ++j) {
	// 		std::cout << this->blobs_[0]->cpu_diff()[i*N_+j] << ",\t";
	// 	}
	// 	std::cout << std::endl;
	// }

	return Dtype(0);
}

INSTANTIATE_CLASS(InnerProductLayerUncorrelated);

}  // namespace caffe
