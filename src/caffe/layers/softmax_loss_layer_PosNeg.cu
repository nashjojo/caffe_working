// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

// Kaixiang Mo
#include <iostream>
#include <typeinfo>
// ~Kaixiang Mo

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayerPosNeg<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 5) << "SoftmaxLoss Layer takes 5 blobs as input, 1:data 2:label 3:weight 4:id 5:negative-weight.";
	CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
	softmax_bottom_vec_.clear();
	softmax_bottom_vec_.push_back(bottom[0]);
	softmax_top_vec_.push_back(&prob_);
	softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);

	// size the same with bottom[0]
	neg_gradient_.reset(
			new Blob<Dtype>(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()));
};

template <typename Dtype>
void SoftmaxWithLossLayerPosNeg<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	// The forward pass computes the softmax prob values.
	softmax_bottom_vec_[0] = bottom[0];
	softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithLossLayerPosNeg<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	// The forward pass computes the softmax prob values.
	softmax_bottom_vec_[0] = bottom[0];
	softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
	
	// int num = bottom[0]->num();
	// int dim = bottom[0]->count() / num;
	// const Dtype* indata = bottom[0]->cpu_data();
	// std::cout << "SoftmaxWithLossLayerPosNeg softmax input" << std::endl; 
	// for (int i = 0; i < 5; ++i) {
	// for (int j = 0; j < std::min(10,dim); ++j ){
		// std::cout << indata[i * dim + j] << "\t";
	// }
	// std::cout << std::endl; 
	// }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayerPosNeg<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	// First, compute the diff
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	Dtype* neg_diff = neg_gradient_->mutable_cpu_diff();

	const Dtype* prob_data = prob_.cpu_data();

	memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
	memcpy(neg_diff, prob_data, sizeof(Dtype) * prob_.count());

	const Dtype* label = (*bottom)[1]->cpu_data();
	const Dtype* weight = (*bottom)[2]->cpu_data();
	const Dtype* neg_weight = (*bottom)[4]->cpu_data();
	//const long int* id = reinterpret_cast<const long int*>((*bottom)[3]->cpu_data()); // Kaixiang MO, 25th April, 2014
	
	int num = prob_.num();
	int dim = prob_.count() / num;
	Dtype loss = 0;
	Dtype weight_sum = 0;

	// std::cout << "SoftmaxWithLossLayerPosNeg prob input" << std::endl; 
	// for (int i = 0; i < 5; ++i) {
	// 	for (int j = 0; j < std::min(10,dim); ++j ){
	// 		std::cout << prob_data[i * dim + j] << "\t";
	// 	}
	// 	std::cout << std::endl; 
	// }
	
	for (int i = 0; i < num; ++i) {
		// std::cout << "id\t" << id[i] << "\tweight\t" << weight[i] << "\tneg-weight\t" << neg_weight[i] << std::endl;
		// positive weight
		weight_sum += weight[i];
		bottom_diff[i * dim + 1] -= 1;

		// print the lines
		// std::cout << i <<" "<< id[i] << " before mul weight " << weight[i] << std::endl;
		// for (int j = 0; j < std::min(10,dim); ++j ){
		// 	std::cout << bottom_diff[i * dim + j] << "\t";
		// }
		// std::cout << std::endl; 
		
		caffe_scal(dim, weight[i], bottom_diff + i * dim);
		loss += -log(max(prob_data[i * dim + 1], FLT_MIN)) * weight[i];

		// std::cout << i <<" "<< id[i] << " after mul weight " << weight[i] << std::endl;
		// for (int j = 0; j < std::min(10,dim); ++j ){
		// 	std::cout << bottom_diff[i * dim + j] << "\t";
		// }
		// std::cout << std::endl; 

		//negative weight
		weight_sum += neg_weight[i];
		neg_diff[i * dim + 0] -= 1;
		
		// std::cout << i <<" "<< id[i] << " before mul neg_weight " << neg_weight[i] << std::endl;
		// for (int j = 0; j < std::min(10,dim); ++j ){
		// 	std::cout << neg_diff[i * dim + j] << "\t";
		// }
		// std::cout << std::endl; 

		caffe_scal(dim, neg_weight[i], neg_diff + i * dim);
		loss += -log(max(prob_data[i * dim + 0], FLT_MIN)) * neg_weight[i];

		// std::cout << i <<" "<< id[i] << " after mul neg_weight " << neg_weight[i] << std::endl;
		// for (int j = 0; j < std::min(10,dim); ++j ){
		// 	std::cout << neg_diff[i * dim + j] << "\t";
		// }
		// std::cout << std::endl; 

	}
	// addint negative gradient to bottom_diff
	caffe_axpy(prob_.count(), Dtype(1), neg_diff, bottom_diff);

	// std::cout << "SoftmaxWithLossLayerPosNeg diff result" << std::endl; 
	// for (int i = 0; i < 5; ++i) {
	// 	for (int j = 0; j < std::min(10,dim); ++j ){
	// 		std::cout << bottom_diff[i * dim + j] << "\t";
	// 	}
	// 	std::cout << std::endl; 
	// }

	// normalize using total weight
	caffe_scal(prob_.count(), Dtype(1) / weight_sum, bottom_diff);

	return loss / weight_sum;
}

template <typename Dtype>
Dtype SoftmaxWithLossLayerPosNeg<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	// TODO(Yangqing): implement the GPU version of softmax.
	return Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(SoftmaxWithLossLayerPosNeg);


}  // namespace caffe
