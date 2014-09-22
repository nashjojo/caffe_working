// Copyright 2013 Yangqing Jia
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

// Kaixiang MO
#include <iostream>
#include <iomanip>
// ~Kaixiang Mo

using namespace std;
using std::max;

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
	CHECK_EQ(top->size(), 0) << "Loss Layer takes no output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
};


template <typename Dtype>
Dtype MultinomialLogisticLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	const Dtype* bottom_data = (*bottom)[0]->cpu_data();
	const Dtype* bottom_label = (*bottom)[1]->cpu_data();
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	int num = (*bottom)[0]->num();
	int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
	memset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count());
	Dtype loss = 0;
	for (int i = 0; i < num; ++i) {
		int label = static_cast<int>(bottom_label[i]);
		Dtype prob = max(bottom_data[i * dim + label], kLOG_THRESHOLD);
		loss -= log(prob);
		bottom_diff[i * dim + label] = - 1. / prob / num;
	}
	return loss / num;
}

// TODO: implement the GPU version for multinomial loss


template <typename Dtype>
void InfogainLossLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
	CHECK_EQ(top->size(), 0) << "Loss Layer takes no output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	BlobProto blob_proto;
	ReadProtoFromBinaryFile(this->layer_param_.source(), &blob_proto);
	infogain_.FromProto(blob_proto);
	CHECK_EQ(infogain_.num(), 1);
	CHECK_EQ(infogain_.channels(), 1);
	CHECK_EQ(infogain_.height(), infogain_.width());
};


template <typename Dtype>
Dtype InfogainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	const Dtype* bottom_data = (*bottom)[0]->cpu_data();
	const Dtype* bottom_label = (*bottom)[1]->cpu_data();
	const Dtype* infogain_mat = infogain_.cpu_data();
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	int num = (*bottom)[0]->num();
	int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
	CHECK_EQ(infogain_.height(), dim);
	Dtype loss = 0;
	for (int i = 0; i < num; ++i) {
		int label = static_cast<int>(bottom_label[i]);
		for (int j = 0; j < dim; ++j) {
			Dtype prob = max(bottom_data[i * dim + j], kLOG_THRESHOLD);
			loss -= infogain_mat[label * dim + j] * log(prob);
			bottom_diff[i * dim + j] = - infogain_mat[label * dim + j] / prob / num;
		}
	}
	return loss / num;
}


template <typename Dtype>
void EuclideanLossLayer<Dtype>::SetUp(
	const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
	CHECK_EQ(top->size(), 0) << "Loss Layer takes no as output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
	CHECK_EQ(bottom[0]->height(), bottom[1]->height());
	CHECK_EQ(bottom[0]->width(), bottom[1]->width());
	difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
			bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
Dtype EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	int count = (*bottom)[0]->count();
	int num = (*bottom)[0]->num();
	caffe_sub(count, (*bottom)[0]->cpu_data(), (*bottom)[1]->cpu_data(),
			difference_.mutable_cpu_data());
	Dtype loss = caffe_cpu_dot(
			count, difference_.cpu_data(), difference_.cpu_data()) / num / Dtype(2);
	// Compute the gradient
	caffe_axpby(count, Dtype(1) / num, difference_.cpu_data(), Dtype(0),
			(*bottom)[0]->mutable_cpu_diff());
	return loss;
}

template <typename Dtype>
void AccuracyLayer<Dtype>::SetUp(
	const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
	CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	(*top)[0]->Reshape(1, 2, 1, 1);
}

/*template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu_test(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	fstream out("/data/nwangab/test/test_dump.txt",ios::out|ios::app);
	Dtype accuracy = 0;
	Dtype logprob = 0;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	Transforms types;
	ReadProtoFromTextFile(this->layer_param_.trans_type(),&types);
	for (int i = 0; i < num; i+=types.transformtype_size()) {
		// Accuracy
		out<<"new vector"<<endl;
		vector<Dtype>val(dim,0);
		for(int j=0;j<types.transformtype_size();++j){
	for(int k=0;k<dim;++k){
		val[k]+=bottom_data[(i+j)*dim+k];
		out<<bottom_data[(i+j)*dim+k]<<" ";
	}
	out<<endl;
		} 
		out<<endl<<endl;
		Dtype maxval = -FLT_MAX;
		int max_id = 0;
		for (int j = 0; j < dim; ++j) {
			if (val[j]> maxval) {
				maxval = val[j];
				max_id = j;
			}
		}
		if (max_id == (int)bottom_label[i]) {
			++accuracy;
		}
	 // Dtype prob = max(bottom_data[i * dim + (int)bottom_label[i]], kLOG_THRESHOLD);
		 Dtype prob = max(val[(int)bottom_label[i]]/(Dtype)types.transformtype_size(), kLOG_THRESHOLD);
		 logprob -= log(prob);
	}
	// LOG(INFO) << "Accuracy: " << accuracy;
	(*top)[0]->mutable_cpu_data()[0] = accuracy / (num/(Dtype)types.transformtype_size());
	(*top)[0]->mutable_cpu_data()[1] = logprob / (num/(Dtype)types.transformtype_size());
	out.close();
}
*/
template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	ofstream out,out2;
	if(this->layer_param_.has_data_dump() && !this->layer_param_.has_label_dump()){
		LOG(FATAL)<<"data dump and label dump files should exist at the same time.";
	}
	if(!this->layer_param_.has_data_dump() && this->layer_param_.has_label_dump()){
		LOG(FATAL)<<"data dump and label dump files should exist at the same time.";
	}
	if(this->layer_param_.has_data_dump()){
		out.open(this->layer_param_.data_dump().c_str(),ios::out|ios::app);
	}
	if(this->layer_param_.has_label_dump()){
		out2.open(this->layer_param_.label_dump().c_str(),ios::out|ios::app);
	}
	Dtype accuracy = 0;
	Dtype logprob = 0;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	for (int i = 0; i < num; i++) {
		// Accuracy
	//  out<<"new vector"<<endl;
		Dtype maxval = -FLT_MAX;
		Dtype eps = 1e-6;
		int max_id = 0;
		for (int j = 0; j < dim; ++j) {  
			if(Caffe::phase()==Caffe::TEST){
				if(this->layer_param_.has_data_dump()){
					if(bottom_data[i*dim+j]>=eps){
						out<<bottom_data[i*dim+j]<<" ";
					} else {
						out<<0<<" ";
					}
					if(j==dim-1)
						out<<endl;
				}
			}
			if (bottom_data[i*dim+j]> maxval) {
				maxval = bottom_data[i*dim+j];
				max_id = j;
			}
		}
		if(Caffe::phase()==Caffe::TEST){
			if(this->layer_param_.has_label_dump()){
				out2<<(int)bottom_label[i]<<endl;
			}
		}
		if (max_id == (int)bottom_label[i]) {
			++accuracy;
		}
		Dtype prob = max(bottom_data[i * dim + (int)bottom_label[i]], kLOG_THRESHOLD);
	 //  Dtype prob = max(val[(int)bottom_label[i]], KLOG_THRESHOLD);
		 logprob -= log(prob);
	}
	// LOG(INFO) << "Accuracy: " << accuracy;
	(*top)[0]->mutable_cpu_data()[0] = accuracy / num;
	(*top)[0]->mutable_cpu_data()[1] = logprob / num;
	if(this->layer_param_.has_data_dump()){
		out.close();
	}
	if(this->layer_param_.has_label_dump()){
		out2.close();
	}
}

// Kaixiang Mo, 25th April, 2014
template <typename Dtype>
void DumpLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 4) << "Auc Layer takes 4 blobs as input. data, label, weight, id";
	CHECK_EQ(top->size(), 1) << "Auc Layer takes 1 output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	(*top)[0]->Reshape(1, 2, 1, 1);
}

template <typename Dtype>
void DumpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	ofstream out;
	if(!this->layer_param_.has_data_dump()){
		LOG(FATAL)<<"data dump file do not exist.";
		(*top)[0]->mutable_cpu_data()[0] = 0;
		(*top)[0]->mutable_cpu_data()[1] = 0;
	return;
	}
	out.open(this->layer_param_.data_dump().c_str(),ios::out|ios::app);
	Dtype accuracy = 0;
	Dtype logprob = 0;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	const Dtype* bottom_weight = bottom[2]->cpu_data();
	const int* bottom_id = reinterpret_cast<const int*>((bottom)[3]->cpu_data());
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	
	// sample 10 top_data and print out
	// const Dtype* bottom_data1 = bottom[0]->cpu_data();
	// std::cout << this->layer_param_.name() << " output" << std::endl;
	// for (int i = 0; i < 5; i++) {
	//   std::cout << std::setw(10) << bottom_data1[i] <<"\t";
	// }
	// std::cout << std::endl;
	
	for (int i = 0; i < num; i++) {
	// max value
		Dtype maxval = -FLT_MAX;
		Dtype eps = 1e-6;
		int max_id = 0;
	// data
		for (int j = 0; j < dim; ++j) {
		//std::cout<<bottom_data[i*dim+j]<<" ";
			if (bottom_data[i*dim+j] >= eps) {
				out<<bottom_data[i*dim+j]<<" ";
		//std::cout<<bottom_data[i*dim+j]<<" ";
			} else {
				out<<0<<" ";
		//std::cout<<0<<" ";
			}
			if (bottom_data[i*dim+j]> maxval) {
				maxval = bottom_data[i*dim+j];
				max_id = j;
			}
		}
	// label, weight, adid
		out << (int)bottom_label[i] <<" "<< (int)bottom_weight[i] <<" "<< (int)bottom_id[i] << endl;
	//std::cout << (int)bottom_label[i] <<" "<< (int)bottom_weight[i] <<" "<< (int)bottom_id[i] << std::endl;
		if ( max_id == (int)bottom_label[i] ) {
			++accuracy;
		}
		Dtype prob = max(bottom_data[i * dim + (int)bottom_label[i]], kLOG_THRESHOLD);
		logprob -= log(prob);
	}
	// LOG(INFO) << "Accuracy: " << accuracy;
	(*top)[0]->mutable_cpu_data()[0] = accuracy / num;
	(*top)[0]->mutable_cpu_data()[1] = logprob / num;
	out.close();
}
// ~Kaixiang Mo, 25th April, 2014

// Kaixiang Mo, 30th June, 2014
template <typename Dtype>
void DumpLayerPosNeg<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 5) << "DumpLayerPosNeg Layer takes 5 blobs as input. 1:data, 2:label, 3:weight, 4:id, 5:neg-weight";
	CHECK_EQ(top->size(), 1) << "DumpLayerPosNeg Layer takes 1 dummy output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	(*top)[0]->Reshape(1, 2, 1, 1);
}

template <typename Dtype>
void DumpLayerPosNeg<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	ofstream out;
	if(!this->layer_param_.has_data_dump()){
		LOG(FATAL)<<"data dump file do not exist.";
		(*top)[0]->mutable_cpu_data()[0] = 0;
		(*top)[0]->mutable_cpu_data()[1] = 0;
	return;
	}
	out.open(this->layer_param_.data_dump().c_str(),ios::out|ios::app);
	Dtype accuracy = 0;
	Dtype logprob = 0;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	const Dtype* bottom_weight = bottom[2]->cpu_data();
	const Dtype* bottom_neg_weight = bottom[4]->cpu_data();
	const int* bottom_id = reinterpret_cast<const int*>((bottom)[3]->cpu_data());
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	CHECK_EQ(dim, 2) << "DumpLayerPosNeg require exactly 2 output classes";

	
	// sample 10 top_data and print out
	// const Dtype* bottom_data1 = bottom[0]->cpu_data();
	// std::cout << this->layer_param_.name() << " output" << std::endl;
	// for (int i = 0; i < 5; i++) {
	//   std::cout << std::setw(10) << bottom_data1[i] <<"\t";
	// }
	// std::cout << std::endl;
	
	for (int i = 0; i < num; i++) {
		// max value
		Dtype maxval = -FLT_MAX;
		Dtype eps = 1e-6;
		int max_id = 0;
		Dtype out_prob[2];
		
		// data
		for (int j = 0; j < dim; ++j) {
		//std::cout<<bottom_data[i*dim+j]<<" ";
			if (bottom_data[i*dim+j] >= eps) {
				out_prob[j] = bottom_data[i*dim+j];
				// out<<bottom_data[i*dim+j]<<" ";
				//std::cout<<bottom_data[i*dim+j]<<" ";
			} else {
				out_prob[j] = 0;
				// out<<0<<" ";
				//std::cout<<0<<" ";
			}
			if (bottom_data[i*dim+j] > maxval) {
				maxval = bottom_data[i*dim+j];
				max_id = j;
			}
		}
		// label, weight, adid
		out << out_prob[0] << " " << out_prob[1] << " " << 1 <<" "<< (int)bottom_weight[i] <<" "<< (int)bottom_id[i] << endl;
		out << out_prob[0] << " " << out_prob[1] << " " << 0 <<" "<< (int)bottom_neg_weight[i] <<" "<< (int)bottom_id[i] << endl;
		//std::cout << (int)bottom_label[i] <<" "<< (int)bottom_weight[i] <<" "<< (int)bottom_id[i] << std::endl;
		if ( max_id == (int)bottom_label[i] ) {
			++accuracy;
		}
		Dtype prob = max(bottom_data[i * dim + (int)bottom_label[i]], kLOG_THRESHOLD);
		logprob -= log(prob);
	}
	// LOG(INFO) << "Accuracy: " << accuracy;
	(*top)[0]->mutable_cpu_data()[0] = accuracy / num;
	(*top)[0]->mutable_cpu_data()[1] = logprob / num;
	out.close();
}
// ~Kaixiang Mo, 30th June, 2014

INSTANTIATE_CLASS(MultinomialLogisticLossLayer);
INSTANTIATE_CLASS(InfogainLossLayer);
INSTANTIATE_CLASS(EuclideanLossLayer);
INSTANTIATE_CLASS(AccuracyLayer);
INSTANTIATE_CLASS(DumpLayer); // Kaixiang Mo, 25th April, 2014
INSTANTIATE_CLASS(DumpLayerPosNeg); // Kaixiang MO, 30th June, 2014

}  // namespace caffe
