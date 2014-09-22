// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h> // for sqrt

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>

#include <typeinfo>

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayerWeightedExtFeature<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 5) << "SoftmaxLoss Layer takes 5 blobs as input, 3rd blob is the weight blob, 4th blob is the id, 5th blob is the extra feature";
	CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
	softmax_bottom_vec_.clear();
	softmax_bottom_vec_.push_back(bottom[0]);
	softmax_top_vec_.push_back(&prob_);
	softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
};

template <typename Dtype>
void SoftmaxWithLossLayerWeightedExtFeature<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	// The forward pass computes the softmax prob values.
	softmax_bottom_vec_[0] = bottom[0];
	softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithLossLayerWeightedExtFeature<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	// The forward pass computes the softmax prob values.
	softmax_bottom_vec_[0] = bottom[0];
	softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype SoftmaxWithLossLayerWeightedExtFeature<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	// First, compute the diff
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	const Dtype* prob_data = prob_.cpu_data();
	memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
	const Dtype* label = (*bottom)[1]->cpu_data();
	const Dtype* weight = (*bottom)[2]->cpu_data();
	//const int* id = reinterpret_cast<int*>((*bottom)[3]->cpu_data()); // Kaixiang MO, 25th April, 2014
	const Dtype* extfeature = (*bottom)[4]->cpu_data();
	
	//const int num_extfeature = this->layer_param_.num_extfeature();
	// No need to specify
	const int num_extfeature = (*bottom)[4]->channels();
	// LOG(INFO) << "num_extfeature is " << num_extfeature;
	// the factor for penalize correlation
	const Dtype lambda = this->layer_param_.covar_factor();
	// LOG(INFO) << "lambda is " << lambda; 
	int num = prob_.num();
	int dim = prob_.count() / num;
	Dtype loss = 0;
	Dtype weight_sum = 0;
	std::vector<std::vector<Dtype> > covar;
	std::vector<std::vector<Dtype> > feature;
	std::vector<Dtype> f_mean;
	std::vector<std::vector<Dtype> > prob;
	std::vector<Dtype> p_mean;

	std::vector<Dtype> instance_weight;
	Dtype instance_weight_sum = Dtype(0.);
	for (int j = 0; j < num; j++) {
		instance_weight.push_back(weight[j]);
		instance_weight_sum += instance_weight[j];
	}
	

	// LOG(INFO) << "Calculating Correlation";
	// calculate the mean of prob.
	for (int j = 0; j < dim; j++) {
		Dtype p_mean_temp = 0;
		std::vector<Dtype> temp;
		for (int k = 0; k < num; k++) {
			p_mean_temp = p_mean_temp + prob_data[k*dim + j] * instance_weight[k];
			temp.push_back(prob_data[k*dim + j]);
		}
		prob.push_back(temp);
		p_mean.push_back(p_mean_temp / instance_weight_sum);
		// if (p_mean_temp/num == 0) {
			// std::cout<< "\tProb " << j <<" is all 0"<< std::endl;
		// }
	}
	
	// calculate the mean of the extra-features.
	for (int j = 0; j < num_extfeature; j++) {
		Dtype f_mean_temp = 0;
		std::vector<Dtype> temp;
		for (int k = 0; k < num; k++) {
			f_mean_temp = f_mean_temp + extfeature[k*num_extfeature + j] * instance_weight[k];
			temp.push_back(extfeature[k*num_extfeature + j]);
		}
		feature.push_back(temp);
		f_mean.push_back(f_mean_temp / instance_weight_sum);
		// if (f_mean_temp/num == 0) {
			// std::cout<< "\tFeature " << j <<" is all 0"<< std::endl;
		// }
	}
	
	// the correlation of all pair of features.
	// std::cout << "correlation" << std::endl;
	for (int j = 0; j < dim; j++) {
		std::vector<Dtype> temp;
		for (int k = 0; k < num_extfeature; k++) {
			Dtype co = Correlation( prob[j], p_mean[j], feature[k], f_mean[k], instance_weight );
			temp.push_back( co ); // no need to normalized by ( co / Dtype(num) );
			// std::cout << j <<" "<< k <<" "<< co <<" "<< co / Dtype(num) << std::endl;
			// LOG(INFO) << j <<" "<< k <<" "<< co <<" "<< co / Dtype(num);
		}
		covar.push_back(temp);
	}
	
	for (int i = 0; i < num; ++i) {
		//std::cout << "ins\t" << i << "\tid type " << typeid(id[i]).name() << "\tid\t" << id[i] << "\tweight\t" << weight[i] << std::endl;
		weight_sum += weight[i];
		
		bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
		// print the extra features.
		// std::cout << i <<" "<< id[i] << " ext_features "<< std::endl;
		// for (int j = 0; j < num_extfeature; j++) {
			// std::cout<< extfeature[i * num_extfeature + j] <<" ";
		// }
		// std::cout << std::endl;
	
		// print the lines
		// std::cout << i <<" "<< id[i] << " prob"<< std::endl;
		// for (int j = 0; j < std::min(10,dim); ++j ){
			// std::cout << prob_data[i * dim + j] << "\t";
		// }
		// std::cout << std::endl; 
		
		// penalize correlation of different feature
		// std::cout << "Penalize correlation" << std::endl;
		for (int j = 0; j < dim; j++) {
			for (int k = 0; k < num_extfeature; k++) {
				bottom_diff[i * dim + j] += lambda*covar[j][k]*(feature[k][i]-f_mean[k]) / num_extfeature;
			}
		}
	
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
Dtype SoftmaxWithLossLayerWeightedExtFeature<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	// TODO(Yangqing): implement the GPU version of softmax.
	return Backward_cpu(top, propagate_down, bottom);
}

template <typename Dtype>
Dtype SoftmaxWithLossLayerWeightedExtFeature<Dtype>::Correlation(const vector<Dtype>& feature_1, Dtype mean_1, 
		const vector<Dtype>& feature_2, Dtype mean_2, std::vector<Dtype> weight) {
	// TODO(Kaixiang Mo): implement the correlation
	CHECK_EQ(feature_1.size(), feature_2.size()) << "Correlation: features have different length!";
	// std::cout << "mean_1 " << mean_1 << " mean_2 " << mean_2 << std::endl;
	Dtype covar = Dtype(0.);
	Dtype var_1 = Dtype(0.);
	Dtype var_2 = Dtype(0.);
	for (int i = 0; i < feature_1.size(); i++) {
		covar += (feature_1[i]-mean_1) * (feature_2[i]-mean_2) * weight[i];
	// std::cout << i <<" "<< (feature_1[i]-mean_1) <<" "<< (feature_2[i]-mean_2) << std::endl;
		var_1 += (feature_1[i]-mean_1) * (feature_1[i]-mean_1) * weight[i];
		var_2 += (feature_2[i]-mean_2) * (feature_2[i]-mean_2) * weight[i];
	}
	// std::cout <<"Correlation return "<< covar << std::endl;
	return covar / (sqrt(var_1) * sqrt(var_2));
}

INSTANTIATE_CLASS(SoftmaxWithLossLayerWeightedExtFeature);


}  // namespace caffe
