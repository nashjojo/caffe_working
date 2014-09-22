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
void InnerProductLayerExtFeature<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "IP Layer takes 2 blobs as input, 2nd is the extra features.";
	CHECK_EQ(top->size(), 1) << "IP Layer takes a single blob as output.";
	const int num_output = this->layer_param_.num_output();
	num_extfeature = bottom[1]->channels(); // is bottom[1] initialized yet?
	// check not 0
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
			this->blobs_.resize(3); // adding one for extra feature
		} else {
			this->blobs_.resize(2); 
		}
		// Intialize the weight
		this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
		this->blobs_[1].reset(new Blob<Dtype>(1, 1, N_, num_extfeature));
		// fill the weights
		shared_ptr<Filler<Dtype> > weight_filler(
				GetFiller<Dtype>(this->layer_param_.weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
		weight_filler->Fill(this->blobs_[1].get());
	
		// If necessary, intiialize and fill the bias term
		if (biasterm_) {
			this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, N_));
			shared_ptr<Filler<Dtype> > bias_filler(
					GetFiller<Dtype>(this->layer_param_.bias_filler()));
			bias_filler->Fill(this->blobs_[2].get());
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
};

template <typename Dtype>
void InnerProductLayerExtFeature<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* extra_data = bottom[1]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const Dtype* weight = this->blobs_[0]->cpu_data();
	const Dtype* extra_weight = this->blobs_[1]->cpu_data();
	
	// main feature
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, weight, (Dtype)0., top_data);
	// calculating extra feature, adding to the result
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, num_extfeature, (Dtype)1.,
			extra_data, extra_weight, (Dtype)1., top_data);
	if (biasterm_) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
				this->blobs_[2]->cpu_data(), (Dtype)1., top_data);
	}
}

template <typename Dtype>
Dtype InnerProductLayerExtFeature<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = (*bottom)[0]->cpu_data();
	const Dtype* extra_data = (*bottom)[1]->cpu_data();
	
	// Gradient with respect to weight
	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
	// Gradient with respect to extra weight of extra feature
	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, num_extfeature, M_, (Dtype)1.,
			top_diff, extra_data, (Dtype)0., this->blobs_[1]->mutable_cpu_diff());
	if (biasterm_) {
		// Gradient with respect to bias
		caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
				reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)0.,
				this->blobs_[2]->mutable_cpu_diff());
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
void InnerProductLayerExtFeature<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* extra_data = bottom[1]->gpu_data();
	Dtype* top_data = (*top)[0]->mutable_gpu_data();
	const Dtype* weight = this->blobs_[0]->gpu_data(); // #output * #input, need transpose
	const Dtype* extra_weight = this->blobs_[1]->gpu_data();
	
	// main feature
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
			bottom_data, weight, (Dtype)0., top_data);

	// sample 10 top_data and print out
	// const Dtype* top_data1 = (*top)[0]->cpu_data();
	// std::cout << "top_data before add" << std::endl;
	// for (int i = 0; i < 5; i++) {
	// std::cout << std::setw(10) << top_data1[i] <<"\t";
	// }
	// std::cout << std::endl;
	
	// print all instance
	// const Dtype* extra_data1 = bottom[1]->cpu_data();
	// int batchsize = bottom[1]->num();
	// std::cout << "extra_data batch_size " << batchsize << " num_extfeature " << num_extfeature << std::endl;
	// for (int itemid = 0; itemid < 5; itemid++) {
		// std::cout << "ins" << itemid << std::endl;
		// for (int i = 0; i < num_extfeature; i++) {
		// std::cout << std::setw(10) << extra_data1[itemid * num_extfeature + i] <<"\t";
		// }
	// std::cout << std::endl;
	// }
	
	// sample 10 extra_weight and print out
	// const Dtype* extra_weight1 = this->blobs_[1]->cpu_data();
	// std::cout << "extra weight" << std::endl;
	// for (int i = 0; i < 5; i++) {
	// std::cout << std::setw(10) << extra_weight1[i] <<"\t";
	// }
	// std::cout << std::endl;

	
	// calculating extra feature, adding to the result
	// Dtype* top_data2 = (*top)[0]->mutable_gpu_data();
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, num_extfeature, (Dtype)1.,
			extra_data, extra_weight, (Dtype)1., top_data);
		
	// sample 10 top_data and print out
	// const Dtype* top_data3 = (*top)[0]->cpu_data();
	// std::cout << "top_data after add" << std::endl;
	// for (int i = 0; i < 5; i++) {
	// std::cout << std::setw(10) << top_data3[i] <<"\t";
	// }
	// std::cout << std::endl;

	// Dtype* top_data4 = (*top)[0]->mutable_gpu_data();
	if (biasterm_) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
				reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
				this->blobs_[2]->gpu_data(), (Dtype)1., top_data);
	}
	
	// sample 10 top_data and print out
	// const Dtype* top_data3 = (*top)[0]->cpu_data();
	// std::cout << this->layer_param_.name() << " output" << std::endl;
	// for (int i = 0; i < 5; i++) {
	// std::cout << std::setw(10) << top_data3[i] <<"\t";
	// }
	// std::cout << std::endl;
	
	// test correctness of matrix multiplication
	// unit_test();
}

template <typename Dtype>
Dtype InnerProductLayerExtFeature<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = (*bottom)[0]->gpu_data();
	const Dtype* extra_data = (*bottom)[1]->gpu_data();
	
	// Gradient with respect to weight
	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
			top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
		
	// Gradient with respect to extra weight of extra feature
	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, num_extfeature, M_, (Dtype)1.,
			top_diff, extra_data, (Dtype)0., this->blobs_[1]->mutable_gpu_diff());
	if (biasterm_) {
		// Gradient with respect to bias
		caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
				reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
				(Dtype)0., this->blobs_[2]->mutable_gpu_diff());
	}
	if (propagate_down) {
		// Gradient with respect to bottom data
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
				top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
				(*bottom)[0]->mutable_gpu_diff());
	}
	return Dtype(0);
}

/* // Kaixiang MO: to test the correctness of matrix multiplication
template <typename Dtype>
void InnerProductLayerExtFeature<Dtype>::unit_test() {

	const int m = 5;
	const int n = 4;
	const int k = 4;
	// LOG(INFO) << "unit_test a";
	// shared_ptr<Blob<Dtype> > mx_a;
	// mx_a.reset(new Blob<Dtype>());
	// LOG(INFO) << "unit_test a";
	// mx_a->Reshape(1, 1, 5, 4); // wrong here!
	// LOG(INFO) << "unit_test a";
	// Dtype* mx_a_p = mx_a->mutable_cpu_data();
	Dtype* mx_a_p = new Dtype[m*n];
	
	// LOG(INFO) << "unit_test b";
	// shared_ptr<Blob<Dtype> > mx_b;
	// mx_b.reset(new Blob<Dtype>());
	// mx_b->Reshape(1, 1, 4, 4);
	// Dtype* mx_b_p = mx_b->mutable_cpu_data();
	Dtype* mx_b_p = new Dtype[3*4];
	
	
	// LOG(INFO) << "unit_test c";
	// shared_ptr<Blob<Dtype> > mx_c;
	// mx_c.reset(new Blob<Dtype>());
	// mx_c->Reshape(1, 1, 5, 4);
	// Dtype* mx_c_p = mx_c->mutable_cpu_data();
	Dtype* mx_c_p = new Dtype[m*3];
	
	// random assign value to the matrix
	std::cout << "a" << std::endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
		mx_a_p[i*n + j] = Dtype(i*n+j);
		std::cout << mx_a_p[i*n+j] <<"\t";
	}
	std::cout << std::endl;
	}
	std::cout << "b" << std::endl;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
		mx_b_p[i*4 + j] = Dtype(i*4+j);
		std::cout << mx_b_p[i*4+j] <<"\t";
	}
	std::cout << std::endl;
	}
	std::cout << "c" << std::endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < 3; j++) {
		mx_c_p[i*3 + j] = Dtype(i*3+j);
		std::cout << mx_c_p[i*3+j] <<"\t";
	}
	std::cout << std::endl;
	}
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, m, 3, 4, (Dtype)1.,
			mx_a_p, mx_b_p, (Dtype)0., mx_c_p);

	std::cout << "c result" << std::endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < 3; j++) {
		std::cout << mx_c_p[i*3+j] <<"\t";
	}
	std::cout << std::endl;
	}
	
	delete[] mx_a_p;
	delete[] mx_b_p;
	delete[] mx_c_p;
} */

INSTANTIATE_CLASS(InnerProductLayerExtFeature);

}  // namespace caffe
