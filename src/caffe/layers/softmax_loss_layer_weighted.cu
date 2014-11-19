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
void SoftmaxWithLossLayerWeighted<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 4) << "SoftmaxLoss Layer takes 4 blobs as input, 3rd blob is the weight blob, 4th blob is the id.";
	CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
	softmax_bottom_vec_.clear();
	softmax_bottom_vec_.push_back(bottom[0]);
	softmax_top_vec_.push_back(&prob_);
	softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
};

template <typename Dtype>
void SoftmaxWithLossLayerWeighted<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	// The forward pass computes the softmax prob values.
	softmax_bottom_vec_[0] = bottom[0];
	softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithLossLayerWeighted<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	// The forward pass computes the softmax prob values.
	softmax_bottom_vec_[0] = bottom[0];
	softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
	
	// int num = bottom[0]->num();
	// int dim = bottom[0]->count() / num;
	// const Dtype* indata = bottom[0]->cpu_data();
	// std::cout << "SoftmaxWithLossLayerWeighted softmax input" << std::endl; 
	// for (int i = 0; i < 5; ++i) {
	// for (int j = 0; j < std::min(10,dim); ++j ){
		// std::cout << indata[i * dim + j] << "\t";
	// }
	// std::cout << std::endl; 
	// }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayerWeighted<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	// First, compute the diff
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	const Dtype* prob_data = prob_.cpu_data();
	memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
	const Dtype* label = (*bottom)[1]->cpu_data();
	const Dtype* weight = (*bottom)[2]->cpu_data();
	//const long int* id = reinterpret_cast<const long int*>((*bottom)[3]->cpu_data()); // Kaixiang MO, 25th April, 2014
	
	int num = prob_.num();
	int dim = prob_.count() / num;
	Dtype loss = 0;
	Dtype weight_sum = 0;
	
	// std::cout << "SoftmaxWithLossLayerWeighted prob input" << std::endl; 
	// for (int i = 0; i < 5; ++i) {
	// for (int j = 0; j < std::min(10,dim); ++j ){
		// std::cout << prob_data[i * dim + j] << "\t";
	// }
	// std::cout << std::endl; 
	// }
	
	for (int i = 0; i < num; ++i) {
		//std::cout << "ins\t" << i << "\tid type " << typeid(id[i]).name() << "\tid\t" << id[i] << "\tweight\t" << weight[i] << std::endl;
		weight_sum += weight[i];
			
		bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
		
		// print the lines
		// std::cout << i <<" "<< id[i] << " prob"<< std::endl;
		// for (int j = 0; j < std::min(10,dim); ++j ){
			// std::cout << prob_data[i * dim + j] << "\t";
		// }
		// std::cout << std::endl; 
		
		// print the lines
		// std::cout << i <<" "<< id[i] << " before mul weight " << weight[i] << std::endl;
		// for (int j = 0; j < std::min(10,dim); ++j ){
			// std::cout << bottom_diff[i * dim + j] << "\t";
		// }
		// std::cout << std::endl; 
		
		// always does the computation
		caffe_scal(dim, weight[i], bottom_diff + i * dim);
		
		// std::cout << i <<" "<< id[i] << " after mul weight " << weight[i] << std::endl;
		// for (int j = 0; j < std::min(10,dim); ++j ){
			// std::cout << bottom_diff[i * dim + j] << "\t";
		// }
		// std::cout << std::endl; 
	
		loss += -log(max(prob_data[i * dim + static_cast<int>(label[i])], FLT_MIN)) * weight[i];
	}
	// Scale down gradient
	// caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
	// scale down using total weight
	caffe_scal(prob_.count(), Dtype(1) / weight_sum, bottom_diff);
	// std::cout << "sum weight\t" << weight_sum << std::endl;
	return loss / weight_sum; // Kaixiang Mo 28th April, 2014
}

template <typename Dtype>
Dtype SoftmaxWithLossLayerWeighted<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	// TODO(Yangqing): implement the GPU version of softmax.
	return Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(SoftmaxWithLossLayerWeighted);


}  // namespace caffe
