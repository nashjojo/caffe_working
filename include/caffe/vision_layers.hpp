// Copyright 2013 Yangqing Jia

#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <leveldb/db.h>
#include <pthread.h>
#include <cstring>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


// The neuron layer is a specific type of layers that just works on single
// celements.
template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
	explicit NeuronLayer(const LayerParameter& param)
		 : Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
};


template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype> {
 public:
	explicit ReLULayer(const LayerParameter& param)
			: NeuronLayer<Dtype>(param) {}

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};


template <typename Dtype>
class SigmoidLayer : public NeuronLayer<Dtype> {
 public:
	explicit SigmoidLayer(const LayerParameter& param)
			: NeuronLayer<Dtype>(param) {}

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};


template <typename Dtype>
class BNLLLayer : public NeuronLayer<Dtype> {
 public:
	explicit BNLLLayer(const LayerParameter& param)
			: NeuronLayer<Dtype>(param) {}

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};


template <typename Dtype>
class DropoutLayer : public NeuronLayer<Dtype> {
 public:
	explicit DropoutLayer(const LayerParameter& param)
			: NeuronLayer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	shared_ptr<SyncedMemory> rand_vec_;
	float threshold_;
	float scale_;
	unsigned int uint_thres_;
};


template <typename Dtype>
class FlattenLayer : public Layer<Dtype> {
 public:
	explicit FlattenLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	int count_;
};


template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
	explicit InnerProductLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	int M_; // number of instance
	int K_; // number of input dimension
	int N_; // number of output
	bool biasterm_;
	shared_ptr<SyncedMemory> bias_multiplier_;
};

// InnerProduct Layer with extra features inputs
// author: Kaixiang MO
template <typename Dtype>
class InnerProductLayerExtFeature : public Layer<Dtype> {
 public:
	explicit InnerProductLayerExtFeature(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	int M_; // number of instance
	int K_; // number of input dimension
	int N_; // number of output
	bool biasterm_;
	shared_ptr<SyncedMemory> bias_multiplier_;
	
	int num_extfeature; // number of extra feature
	
 private:
	// void unit_test();
};
// ~Inner Product with extra features inputs

// InnerProduct Layer, forcing output uncorrelated to extra features
// author: Kaixiang MO, 12th August, 2014
template <typename Dtype>
class InnerProductLayerUncorrelated : public Layer<Dtype> {
 public:
	explicit InnerProductLayerUncorrelated(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	int M_; // number of instance
	int K_; // number of input dimension
	int N_; // number of output
	bool biasterm_;
	shared_ptr<SyncedMemory> bias_multiplier_;
	
	int num_extfeature; // number of extra feature
	
 private:
	/*
	0: output feature means      (1 * N_)
	1: extfeature means          (1 * num_extfeature)
	2: output feature - mean     (M_ * N_)
	3: extfeature - mean         (M_ * num_extfeature)
	4: covariance matrix	       (num_extfeature * N_)
	*/
	vector<shared_ptr<Blob<Dtype> > > feature_mean_blobs;
	
	// void unit_test();
};
// ~InnerProduct Layer, forcing output uncorrelated to extra features

// InnerProduct Layer, forcing output uncorrelated to extra features, 
// with positive and negative weighted instances
// author: Kaixiang MO, 25th August, 2014
template <typename Dtype>
class InnerProductLayerUncorrelatedPosNeg : public Layer<Dtype> {
 public:
	explicit InnerProductLayerUncorrelatedPosNeg(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	int M_; // number of instance
	int K_; // number of input dimension
	int N_; // number of output
	bool biasterm_;
	shared_ptr<SyncedMemory> bias_multiplier_;
	
	int num_extfeature; // number of extra feature
	
 private:
	/*
	0:	output feature means 			(1 * N_)
	1:	extfeature means 					(1 * num_extfeature)
	2:	(output feature - mean) 	(M_ * N_)
	3:	(extfeature - mean) .* weight 			(M_ * num_extfeature)
	4:	covariance matrix 				(num_extfeature * N_)
	5:	instance weight vector 		(1 * M_)
	6:	instance weight sum 			(1 * 1)
	*/
	vector<shared_ptr<Blob<Dtype> > > feature_mean_blobs;
	
	// void unit_test();
};
// ~InnerProduct Layer, forcing output uncorrelated to extra features

template <typename Dtype>
class PaddingLayer : public Layer<Dtype> {
 public:
	explicit PaddingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	unsigned int PAD_;
	int NUM_;
	int CHANNEL_;
	int HEIGHT_IN_;
	int WIDTH_IN_;
	int HEIGHT_OUT_;
	int WIDTH_OUT_;
};


template <typename Dtype>
class LRNLayer : public Layer<Dtype> {
 public:
	explicit LRNLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	// scale_ stores the intermediate summing results
	Blob<Dtype> scale_;
	int size_;
	int pre_pad_;
	Dtype alpha_;
	Dtype beta_;
	int num_;
	int channels_;
	int height_;
	int width_;
};


template <typename Dtype>
class Im2colLayer : public Layer<Dtype> {
 public:
	explicit Im2colLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	int KSIZE_;
	int STRIDE_;
	int CHANNELS_;
	int HEIGHT_;
	int WIDTH_;
	int PAD_;
};


template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
	explicit PoolingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	int KSIZE_;
	int STRIDE_;
	int CHANNELS_;
	int HEIGHT_;
	int WIDTH_;
	int POOLED_HEIGHT_;
	int POOLED_WIDTH_;
	Blob<float> rand_idx_;
	Blob<float> max_idx_;
};


template <typename Dtype>
class ConvolutionLayer : public Layer<Dtype> {
 public:
	explicit ConvolutionLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	void RenormalizeFilter();

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	Blob<Dtype> col_bob_;

	int KSIZE_;
	int STRIDE_;
	int NUM_;
	int CHANNELS_;
	int HEIGHT_;
	int WIDTH_;
	int NUM_OUTPUT_;
	int GROUP_;
	int PAD_;
	Blob<Dtype> col_buffer_;
	shared_ptr<SyncedMemory> bias_multiplier_;
	bool biasterm_;
	int M_;
	int K_;
	int N_;
};

// Kaixiang Mo, 20th April, 2014
void GetOffAndResolvesize(int height, int width, Resolves & resolves, int & h_off, int & w_off, int & resolvesize);
// ~Kaixiang Mo, 20th April, 2014

void GetWidthAndHeightOff(bool flag, std::string side, int height, int width,
	int len, int & h_off, int & w_off);

template <typename Dtype>
void ConstructLookUp(Dtype* mapping, const float& luminance, const float& contrast);

// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void* DataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
void* DataLayerPrefetchForTest(void* layer_pointer);

template <typename Dtype>
class DataLayer : public Layer<Dtype> {
	// The function used to perform prefetching.
	friend void* DataLayerPrefetch<Dtype>(void* layer_pointer);
	friend void* DataLayerPrefetchForTest<Dtype>(void* layer_pointer);

 public:
	explicit DataLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual ~DataLayer();
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	shared_ptr<leveldb::DB> db_;
	shared_ptr<leveldb::Iterator> iter_;
	int datum_channels_;
	int datum_height_;
	int datum_width_;
	int datum_size_;
	pthread_t thread_;
	shared_ptr<Blob<Dtype> > prefetch_data_;
	shared_ptr<Blob<Dtype> > prefetch_label_;
	Blob<Dtype> data_mean_;
};

// Kaixiang Mo, 20th April, 2014
// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void* DataLayerWeightedPrefetch(void* layer_pointer);

template <typename Dtype>
void* DataLayerWeightedPrefetchForTest(void* layer_pointer);

// Kaixiang Mo, 20th April, 2014
template <typename Dtype>
class DataLayerWeighted : public Layer<Dtype> {
	// The function used to perform prefetching.
	friend void* DataLayerWeightedPrefetch<Dtype>(void* layer_pointer);
	friend void* DataLayerWeightedPrefetchForTest<Dtype>(void* layer_pointer);

 public:
	explicit DataLayerWeighted(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual ~DataLayerWeighted();
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	shared_ptr<leveldb::DB> db_;
	shared_ptr<leveldb::Iterator> iter_;
	int datum_channels_;
	int datum_height_;
	int datum_width_;
	int datum_size_;
	
	pthread_t thread_;
	shared_ptr<Blob<Dtype> > prefetch_data_;
	shared_ptr<Blob<Dtype> > prefetch_label_;
	// instance weight
	shared_ptr<Blob<Dtype> > prefetch_weight_;
	// instance id, Dtype or use int32? precision enough?? 
	shared_ptr<Blob<int> > prefetch_id_;
	// instance existing feature
	shared_ptr<Blob<Dtype> > prefetch_extfeature_;
	Blob<Dtype> data_mean_;
};
// ~Kaixiang Mo, 20th April, 2014

// Kaixiang Mo, 27th June, 2014
// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void* DataLayerPosNegPrefetch(void* layer_pointer);

template <typename Dtype>
void* DataLayerPosNegPrefetchForTest(void* layer_pointer);

// Kaixiang Mo, 27th June, 2014
template <typename Dtype>
class DataLayerPosNeg : public Layer<Dtype> {
	// The function used to perform prefetching.
	friend void* DataLayerPosNegPrefetch<Dtype>(void* layer_pointer);
	friend void* DataLayerPosNegPrefetchForTest<Dtype>(void* layer_pointer);

 public:
	explicit DataLayerPosNeg(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual ~DataLayerPosNeg();
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	shared_ptr<leveldb::DB> db_;
	shared_ptr<leveldb::Iterator> iter_;
	int datum_channels_;
	int datum_height_;
	int datum_width_;
	int datum_size_;
	
	pthread_t thread_;
	shared_ptr<Blob<Dtype> > prefetch_data_;
	shared_ptr<Blob<Dtype> > prefetch_label_;
	// instance positive weight, #click
	shared_ptr<Blob<Dtype> > prefetch_weight_;
	// instance negative weight, #impression
	shared_ptr<Blob<Dtype> > prefetch_neg_weight_;
	// instance id, Dtype or use int32? precision enough?? 
	shared_ptr<Blob<int> > prefetch_id_;
	// instance existing feature
	shared_ptr<Blob<Dtype> > prefetch_extfeature_;
	Blob<Dtype> data_mean_;
};
// ~Kaixiang Mo, 27th June, 2014

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
	explicit SoftmaxLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
		 const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	// sum_multiplier is just used to carry out sum using blas
	Blob<Dtype> sum_multiplier_;
	// scale is an intermediate blob to hold temporary results.
	Blob<Dtype> scale_;
};


template <typename Dtype>
class MultinomialLogisticLossLayer : public Layer<Dtype> {
 public:
	explicit MultinomialLogisticLossLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	// The loss layer will do nothing during forward - all computation are
	// carried out in the backward pass.
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) { return; }
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) { return; }
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	// virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
	//     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class InfogainLossLayer : public Layer<Dtype> {
 public:
	explicit InfogainLossLayer(const LayerParameter& param)
			: Layer<Dtype>(param), infogain_() {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	// The loss layer will do nothing during forward - all computation are
	// carried out in the backward pass.
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) { return; }
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) { return; }
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	// virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
	//     const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	Blob<Dtype> infogain_;
};


// SoftmaxWithLossLayer is a layer that implements softmax and then computes
// the loss - it is preferred over softmax + multinomiallogisticloss in the
// sense that during training, this will produce more numerically stable
// gradients. During testing this layer could be replaced by a softmax layer
// to generate probability outputs.
template <typename Dtype>
class SoftmaxWithLossLayer : public Layer<Dtype> {
 public:
	explicit SoftmaxWithLossLayer(const LayerParameter& param)
			: Layer<Dtype>(param), softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
		 const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
	// prob stores the output probability of the layer.
	Blob<Dtype> prob_;
	// Vector holders to call the underlying softmax layer forward and backward.
	vector<Blob<Dtype>*> softmax_bottom_vec_;
	vector<Blob<Dtype>*> softmax_top_vec_;
};

// Kaixiang Mo, 22th April, 2014
// softmax + multinomiallogisticloss layer, with instance weight
template <typename Dtype>
class SoftmaxWithLossLayerWeighted : public Layer<Dtype> {
 public:
	explicit SoftmaxWithLossLayerWeighted(const LayerParameter& param)
			: Layer<Dtype>(param), softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
		 const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
	// prob stores the output probability of the layer.
	Blob<Dtype> prob_;
	// Vector holders to call the underlying softmax layer forward and backward.
	vector<Blob<Dtype>*> softmax_bottom_vec_;
	vector<Blob<Dtype>*> softmax_top_vec_;
};

// softmax + multinomiallogisticloss layer, with instance weight and handles extra value
template <typename Dtype>
class SoftmaxWithLossLayerWeightedExtFeature : public Layer<Dtype> {
 public:
	explicit SoftmaxWithLossLayerWeightedExtFeature(const LayerParameter& param)
			: Layer<Dtype>(param), softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
		 const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
	// prob stores the output probability of the layer.
	Blob<Dtype> prob_;
	// Vector holders to call the underlying softmax layer forward and backward.
	vector<Blob<Dtype>*> softmax_bottom_vec_;
	vector<Blob<Dtype>*> softmax_top_vec_;
	
 private:
	Dtype Correlation(const vector<Dtype>& feature_1, Dtype mean_1, 
		const vector<Dtype>& feature_2, Dtype mean_2, std::vector<Dtype> weight);
};

// ~Kaixiang Mo, 22th April, 2014

// Kaixiang Mo, 28th June, 2014
// softmax + multinomiallogisticloss layer, with positive and negative instance weight
template <typename Dtype>
class SoftmaxWithLossLayerPosNeg : public Layer<Dtype> {
 public:
	explicit SoftmaxWithLossLayerPosNeg(const LayerParameter& param)
			: Layer<Dtype>(param), softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
		 const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
	// prob stores the output probability of the layer.
	Blob<Dtype> prob_;

	// store the negative gradient, positive gradient in (*bottom)[0]->mutable_cpu_diff();
	shared_ptr<Blob<Dtype> > neg_gradient_;

	// Vector holders to call the underlying softmax layer forward and backward.
	vector<Blob<Dtype>*> softmax_bottom_vec_;
	vector<Blob<Dtype>*> softmax_top_vec_;
};

// ~Kaixiang Mo, 28th June, 2014

// Kaixiang Mo, 31st July, 2014
// softmax + multinomiallogisticloss layer, with positive and negative instance weight and handles extra feature
template <typename Dtype>
class SoftmaxWithLossLayerPosNegExtFeature : public Layer<Dtype> {
 public:
	explicit SoftmaxWithLossLayerPosNegExtFeature(const LayerParameter& param)
			: Layer<Dtype>(param), softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
		 const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
	// prob stores the output probability of the layer.
	Blob<Dtype> prob_;

	// store the negative gradient, positive gradient in (*bottom)[0]->mutable_cpu_diff();
	shared_ptr<Blob<Dtype> > neg_gradient_;

	// Vector holders to call the underlying softmax layer forward and backward.
	vector<Blob<Dtype>*> softmax_bottom_vec_;
	vector<Blob<Dtype>*> softmax_top_vec_;
	
 private:
	Dtype Correlation(const vector<Dtype>& feature_1, Dtype mean_1, 
		const vector<Dtype>& feature_2, Dtype mean_2, std::vector<Dtype> weight);
};
// ~Kaixiang Mo, 31st July, 2014

template <typename Dtype>
class EuclideanLossLayer : public Layer<Dtype> {
 public:
	explicit EuclideanLossLayer(const LayerParameter& param)
			: Layer<Dtype>(param), difference_() {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	// The loss layer will do nothing during forward - all computation are
	// carried out in the backward pass.
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) { return; }
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) { return; }
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	// virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
	//     const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	Blob<Dtype> difference_;
};


template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
 public:
	explicit AccuracyLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	// The accuracy layer should not be used to compute backward operations.
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
		NOT_IMPLEMENTED;
		return Dtype(0.);
	}
};

// Kaixiang Mo, 25th April, 2014
template <typename Dtype>
class DumpLayer : public Layer<Dtype> {
 public:
	explicit DumpLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	// The auc layer should not be used to compute backward operations.
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
		NOT_IMPLEMENTED;
		return Dtype(0.);
	}
};
// ~Kaixiang Mo, 25th April, 2014

// Kaixiang Mo, 30th June, 2014
template <typename Dtype>
class DumpLayerPosNeg : public Layer<Dtype> {
 public:
	explicit DumpLayerPosNeg(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	// The auc layer should not be used to compute backward operations.
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
		NOT_IMPLEMENTED;
		return Dtype(0.);
	}
};
// ~Kaixiang Mo, 30th June, 2014

template <typename Dtype>
class GlobalAvgPoolingLayer : public Layer<Dtype> {
 public:
	explicit GlobalAvgPoolingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	int NUM_;
	int CHANNEL_;
	int HEIGHT_;
	int WIDTH_;
};

// Cascadable Cross Channel Parameteric (CCCP) Pooling Layer is a layer that 
// performs cross feature map parameteric pooling on the feature maps.
// It is equivalent to the mlpconv in the paper "Network In Network".

// The following adds parameters needed for CCCP Pooling Layer.
// The mlpconv has shared mode and unshared mode. Correspondingly the CCCP Layer 
// has only one group and multiple groups. The group parameter can be borrowed 
// from the convolutional layer.
// The number of outputs num_output can also be borrowed from the inner product 
// layer. Which specifies how many nodes come out from each group.
// biasterm, weight_filler and bias_filler can all be borrowed from the inner 
// product layer.
// If the underlying layer of this layer is a convolutional layer, then the feature 
// maps are divided evenly into groups of size "group". Each group of feature map 
// perform cross channel parameteric pooling and generates "num_output" new feature 
// maps. So in total this layer outputs group x num_output feature maps.
// If the underlying layer of this layer is another CCCP, then this layer ignores 
// the grouping of the previous layer and treat all the feature maps from the 
// previous layer as a whole, and redo grouping according to its own group parameter.
// To respect the grouping from previous layer, just set the group parameter to the 
// same value as the previous layer, which is equivalent to the unshared mdoe of 
// mlpconv.
// CCCP Pooling layer is very flexible in creating shared, unshared, partially shared
// types of mlpconv layers.
template <typename Dtype>
class CCCPPoolingLayer : public Layer<Dtype> {
 public:
	explicit CCCPPoolingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
	virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top);
	virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom);
	virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
		 const bool propagate_down, vector<Blob<Dtype>*>* bottom);

	int GROUP_;
	int NUM_OUTPUT_;
	int CHANNEL_;
	int REST_;
	int biasterm_;
	int NUM_;
	shared_ptr<SyncedMemory> bias_multiplier_;
};

// Naiyan Wang, 31st August, 2014
template <typename Dtype>
class SplitLayer : public Layer<Dtype> {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int count_;
};
// ~Naiyan Wang, 31st August, 2014


// Kaixiang MO, 22nd Sept, 2014
/**
 * @brief Takes at least two Blob%s and concatenates them along either the num
 *        or channel dimension, outputting the result.
 */
template <typename Dtype>
class ConcatLayer : public Layer<Dtype> {
 public:
  explicit ConcatLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  // virtual function -> function
  void Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  // virtual inline LayerParameter_LayerType type() const {
  //   return LayerParameter_LayerType_CONCAT;
  // }
  // virtual inline int MinBottomBlobs() const { return 2; }
  // virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 2+)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x_1 @f$
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x_2 @f$
   *   -# ...
   *   - K @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x_K @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (KN \times C \times H \times W) @f$ if concat_dim == 0, or
   *      @f$ (N \times KC \times H \times W) @f$ if concat_dim == 1:
   *      the concatenated output @f$
   *        y = [\begin{array}{cccc} x_1 & x_2 & ... & x_K \end{array}]
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  /**
   * @brief Computes the error gradient w.r.t. the concatenate inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *        respect to the outputs
   *   -# @f$ (KN \times C \times H \times W) @f$ if concat_dim == 0, or
   *      @f$ (N \times KC \times H \times W) @f$ if concat_dim == 1:
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to concatenated outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length K), into which the top gradient
   *        @f$ \frac{\partial E}{\partial y} @f$ is deconcatenated back to the
   *        inputs @f$
   *        \left[ \begin{array}{cccc}
   *          \frac{\partial E}{\partial x_1} &
   *          \frac{\partial E}{\partial x_2} &
   *          ... &
   *          \frac{\partial E}{\partial x_K}
   *        \end{array} \right] =
   *        \frac{\partial E}{\partial y}
   *        @f$
   */
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> col_bob_;
  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int concat_dim_;
};
// ~Kaixiang MO, 22nd Sept, 2014

}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_

