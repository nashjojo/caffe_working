// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <fcntl.h>

#include <string>
#include <vector>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/image_transform.hpp"

// Kaixiang MO, 21 April, 2014
#include <iostream>
#include <fstream>

//#define sideLen 32
using namespace std;
using std::string;

namespace caffe {

template <typename Dtype>
void* DataLayerMultiPrefetchForTest(void* layer_pointer) {
	CHECK(layer_pointer);
	DataLayerMulti<Dtype>* layer = reinterpret_cast<DataLayerMulti<Dtype>*>(layer_pointer);
	CHECK(layer);
	DatumMulti datum;
	CHECK(layer->prefetch_data_);
	Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();

	vector<shared_ptr<Blob<Dtype> > >& top_label = layer->prefetch_label_;
	int datum_num_label_ = layer->datum_num_label_;
	vector<shared_ptr<Blob<Dtype> > >& top_weight = layer->prefetch_weight_;
	int datum_num_weight_ = layer->datum_num_weight_;

	int* top_id = layer->prefetch_id_->mutable_cpu_data();
	const Dtype scale = layer->layer_param_.scale();
	int batchsize = layer->layer_param_.batchsize();
	const int cropsize = layer->layer_param_.cropsize();
	const bool mirror = layer->layer_param_.mirror();

	if (mirror && cropsize == 0) {
		LOG(FATAL) << "Current implementation requires mirror and cropsize to be "
				<< "set at the same time.";
	}
	// datum scales
	const int channels = layer->datum_channels_;
	int height,width; // no constant width and height, desided on the fly
	cv::Mat oriImg, cutImg;
	int h_off, w_off; // cutting offset of image
	int cutSize = 0;
	int transformIdx = 0;

	const int size = layer->datum_size_; // fixed datum size
	const Dtype* mean = layer->data_mean_.cpu_data();

	Transforms types;
	if(layer->layer_param_.has_trans_type()){
		ReadProtoFromTextFile(layer->layer_param_.trans_type(),&types);
		batchsize*=types.transformtype_size();
		LOG(INFO)<<"Each image will be processed "<< types.transformtype_size() << " times.";
	} else {
		LOG(FATAL)<<"Test: No transformation type file.";
	}
	
	for (int itemid = 0; itemid < batchsize; itemid++) {

		// use new datum only if if this is the first transformtype
		transformIdx = itemid % types.transformtype_size();
		if (transformIdx == 0) {
			CHECK(layer->iter_);
			CHECK(layer->iter_->Valid());
			datum.ParseFromString(layer->iter_->value().ToString());
			// read a datum if it have done all transformtype.
		}

		const string& data = datum.data();
		height = datum.height();
		width = datum.width();

		if (cropsize>0) {
			const TransformParameter& transParam=types.transformtype(transformIdx);
			CHECK(data.size()) << "Image cropping only support uint8 data";
			cutSize = transParam.size();
			if (cutSize) {
				//LOG(INFO) << "cutting " <<cutSize<<"*"<<cutSize << " then resizing to " <<cropsize<<"*"<<cropsize;
				getPositioinOffset(transParam.pos(), height, width, cutSize, cutSize, h_off, w_off);
				oriImg = toMat(data, channels, height, width, cutSize, cutSize, h_off, w_off );
				resizeImage(oriImg, cutImg, cropsize, cropsize);
				oriImg = cutImg;
			} else {
				getPositioinOffset(transParam.pos(), height, width, cropsize, cropsize, h_off, w_off);
				oriImg = toMat(data, channels, height, width, cropsize, cropsize, h_off, w_off );
			}
			if (transParam.mirror()) { // this is test, no random
				flipImage(oriImg, cutImg);
				oriImg = cutImg;
			}
			// Do we need to resize for multi-resolution testing? // No first
			// Normal copy
			for (int c = 0; c < channels; ++c) {
				for (int h = 0; h < cropsize; ++h) {
					for (int w = 0; w < cropsize; ++w) {
						top_data[((itemid * channels + c) * cropsize + h) * cropsize + w]
								= (static_cast<Dtype>((uint8_t)oriImg.at<cv::Vec3b>(h,w)[c])
									- mean[(c * cropsize + h ) * cropsize + w]) * scale;
					}
				}
			} // ~ Normal copy
		}
		else {
			// we will prefer to use data() first, and then try float_data()
			if (data.size()) { // flatten the whole image into array.
				for (int j = 0; j < size; ++j) {
					top_data[itemid * size + j] =
							(static_cast<Dtype>((uint8_t)data[j]) - mean[j]) * scale;
				}
			} else {
				for (int j = 0; j < size; ++j) {
					top_data[itemid * size + j] =
							(datum.float_data(j) - mean[j]) * scale;
				}
			}
		} // ~ if(!cropsize)

		// copy other fields
		for (int i = 0; i < datum_num_label_; ++i) {
			top_label[i]->mutable_cpu_data()[itemid] = datum.label(i);
		}
		for (int i = 0; i < datum_num_weight_; ++i) {
			top_weight[i]->mutable_cpu_data()[itemid] = datum.weight(i);
		}
		top_id[itemid] = datum.id();
	
		// if this is the last transform, go to the next iter
		if (transformIdx == types.transformtype_size()-1) {
			layer->iter_->Next();
			if (!layer->iter_->Valid()) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				layer->iter_->SeekToFirst();
			}
		}
	// LOG(INFO)<<"finish transform for itemid "<<itemid;
	} // end for(itemid)
	// LOG(INFO)<<"setup complete.";
	return (void*)NULL;
}

template <typename Dtype>
void* DataLayerMultiPrefetch(void* layer_pointer) {
	CHECK(layer_pointer);
	DataLayerMulti<Dtype>* layer = reinterpret_cast<DataLayerMulti<Dtype>*>(layer_pointer);
	CHECK(layer);
	DatumMulti datum;
	CHECK(layer->prefetch_data_);
	Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();

	vector<shared_ptr<Blob<Dtype> > >& top_label = layer->prefetch_label_;
	int datum_num_label_ = layer->datum_num_label_;
	vector<shared_ptr<Blob<Dtype> > >& top_weight = layer->prefetch_weight_;
	int datum_num_weight_ = layer->datum_num_weight_;

	int* top_id = layer->prefetch_id_->mutable_cpu_data();
	const Dtype scale = layer->layer_param_.scale();
	const int batchsize = layer->layer_param_.batchsize();
	const int cropsize = layer->layer_param_.cropsize();
	const bool mirror = layer->layer_param_.mirror();
	const float luminance_vary = layer->layer_param_.luminance_vary();
	const float contrast_vary = layer->layer_param_.contrast_vary();
	
	if (mirror && cropsize == 0) {
		LOG(FATAL) << "Current implementation requires mirror and cropsize to be "
				<< "set at the same time.";
	}
	// datum scales
	const int channels = layer->datum_channels_;
	const int size = layer->datum_size_; // fixed datum size
	const Dtype* mean = layer->data_mean_.cpu_data();

	float luminance[batchsize];
	float contrast[batchsize];
	if (luminance_vary != 0){
		caffe_vRngGaussian<float>(batchsize, luminance, 0, luminance_vary);
	} else {
		for (int i = 0; i < batchsize; ++i){
			luminance[i] = 0;
		}
	}
	if (contrast_vary != 0){
		caffe_vRngUniform<float>(batchsize, contrast, -contrast_vary, contrast_vary);
	} else {
		for (int i = 0; i < batchsize; ++i){
			contrast[i] = 0;
		}
	}

	for (int itemid = 0; itemid < batchsize; ++itemid) {
		// get a blob
		CHECK(layer->iter_);
		CHECK(layer->iter_->Valid());
		datum.ParseFromString(layer->iter_->value().ToString());
		Resolves resolves;
		if(layer->layer_param_.has_resolve_size()){
				ReadProtoFromTextFile(layer->layer_param_.resolve_size(),&resolves);    
		}
		const string& data = datum.data();
		int height = datum.height();
		int width = datum.width();
		cv::Mat beforeResize,afterResize;
		// Prepare the lookup table for luminance and contrast adjustment.
		// We assume the input range is [0, 255].
		Dtype mapping[256]; 
		caffe::ConstructLookUp<Dtype>(mapping, luminance[itemid], contrast[itemid]);
		if (cropsize) {
			CHECK(data.size()) << "Image cropping only support uint8 data";
			int h_off, w_off;
			int resolvesize=cropsize;
			// We only do random crop when we do training.
			if (Caffe::phase() == Caffe::TRAIN) {
				if(layer->layer_param_.has_resolve_size()) {
					caffe::GetOffAndResolvesize(height,width,resolves,h_off,w_off,resolvesize);
					// LOG(INFO) << "Using resolution " << resolvesize;
				} else {
					h_off = rand() % (height - resolvesize + 1);
					w_off = rand() % (width - resolvesize + 1);
				}
			} else { // Caffe will not call this, because in test phrase it do not go here.
			 // resolvesize=cropsize;
				h_off = (datum.height() - resolvesize) / 2;
				w_off = (datum.width() - resolvesize) / 2;  
			}

			beforeResize = toMat(data, channels, height, width, resolvesize, resolvesize, h_off, w_off );
			resizeImage(beforeResize, afterResize, cropsize, cropsize);

			if (mirror && rand() % 2) {
				// Copy mirrored version
				for (int c = 0; c < channels; ++c) {
					for (int h = 0; h < cropsize; ++h) {
						for (int w = 0; w < cropsize; ++w) {
							int index = ((itemid * channels + c) * cropsize + h) * cropsize
											 + cropsize - 1 - w;
							Dtype temp = mapping[(uint8_t)afterResize.at<cv::Vec3b>(h,w)[c]];
							top_data[index] = (temp - mean[(c * cropsize + h ) * cropsize + w]) * scale;
						}
					}
				}
			} else {
				// Normal copy
				for (int c = 0; c < channels; ++c) {
					for (int h = 0; h < cropsize; ++h) {
						for (int w = 0; w < cropsize; ++w) {
							int index = ((itemid * channels + c) * cropsize + h) * cropsize + w;
							Dtype temp = mapping[(uint8_t)afterResize.at<cv::Vec3b>(h,w)[c]];
							top_data[index] = (temp - mean[(c * cropsize + h ) * cropsize + w]) * scale;
						}
					}
				}
			}
		} else {
			// we will prefer to use data() first, and then try float_data()
			if (data.size()) {
				for (int j = 0; j < size; ++j) {
					int index = itemid * size + j;
					top_data[index] = ((mapping[(uint8_t)data[j]]) - mean[j]) * scale;
				}
			} else {
				for (int j = 0; j < size; ++j) {
					int index = itemid * size + j;
					top_data[index] = ((mapping[(uint8_t)datum.float_data(j)]) - mean[j]) * scale;
				}
			}
		}
		// copy other fields here
		for (int i = 0; i < datum_num_label_; ++i) {
			top_label[i]->mutable_cpu_data()[itemid] = datum.label(i);
		}
		for (int i = 0; i < datum_num_weight_; ++i) {
			top_weight[i]->mutable_cpu_data()[itemid] = datum.weight(i);
		}
		top_id[itemid] = datum.id();

		// go to the next iter
		layer->iter_->Next();
		if (!layer->iter_->Valid()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			layer->iter_->SeekToFirst();
		}
	}

	return (void*)NULL;
}

template <typename Dtype>
DataLayerMulti<Dtype>::~DataLayerMulti<Dtype>() {
	// Finally, join the thread
	CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
void DataLayerMulti<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	LOG(INFO)<<"DataLayerMulti::SetUp running";
	CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
	// 3rd is weight blob, 4rd is id blob, 5th blob is extra feature.
	CHECK_GE(top->size(), 4) << "Data Layer takes more than 4 blobs as output, 1:img, 2:label, 3:weight, 4:instance-id";
	int top_id = 0;
	// Initialize the leveldb
	leveldb::DB* db_temp;
	leveldb::Options options;
	options.create_if_missing = false;
	LOG(INFO) << "Opening leveldb " << this->layer_param_.source();
	leveldb::Status status = leveldb::DB::Open(
			options, this->layer_param_.source(), &db_temp);
	LOG(INFO)<<"open complete";
	CHECK(status.ok()) << "Failed to open leveldb "
			<< this->layer_param_.source() << std::endl << status.ToString();
	db_.reset(db_temp);
	//leveldb::Iterator* itr;
	iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
	iter_->SeekToFirst();
	LOG(INFO)<<"seek first complete";
	// Check if we would need to randomly skip a few data points
	if (this->layer_param_.rand_skip()) {
		unsigned int skip = rand() % this->layer_param_.rand_skip(); // good
		LOG(INFO) << "Skipping first " << skip << " data points.";
		while (skip-- > 0) {
			iter_->Next();
			if (!iter_->Valid()) {
				iter_->SeekToFirst();
			}
		}
	}
	// Read a data point, and use it to initialize the top blob.
	DatumMulti datum;
	datum.ParseFromString(iter_->value().ToString());

	// image
	// LOG(INFO)<<"parse first complete, height " << datum.height() << " width " << datum.width();
	int cropsize = this->layer_param_.cropsize();
	Transforms types;
	int delta=1;

	if (Caffe::phase() == Caffe::TEST) {
		if (this->layer_param_.has_trans_type()) {
			ReadProtoFromTextFile(this->layer_param_.trans_type(),&types);
			delta*=types.transformtype_size();
			LOG(INFO) << "Each image will be transformed " << delta << " times";
		} else {
			LOG(FATAL)<<"Test: No transformation type file.";
		}
	}

	if (cropsize > 0) {
		(*top)[0]->Reshape(
				this->layer_param_.batchsize()*delta, datum.channels(), cropsize, cropsize);
		prefetch_data_.reset(new Blob<Dtype>(
				this->layer_param_.batchsize()*delta, datum.channels(), cropsize, cropsize));
	} else {
		(*top)[0]->Reshape(
				this->layer_param_.batchsize()*delta, datum.channels(), datum.height(),
				datum.width());
		prefetch_data_.reset(new Blob<Dtype>(
				this->layer_param_.batchsize()*delta, datum.channels(), datum.height(),
				datum.width()));
	}
	LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
			<< (*top)[0]->channels() << "," << (*top)[0]->height() << ","
			<< (*top)[0]->width();
	top_id ++;
	
	// label
	// (*top)[1]->Reshape(this->layer_param_.batchsize()*delta, 1, 1, 1);
	// prefetch_label_.reset(
	// 		new Blob<Dtype>(this->layer_param_.batchsize()*delta, 1, 1, 1));

	datum_num_label_ = datum.label_size();
	// LOG(INFO) << "Setup: datum_num_label_ " << datum_num_label_;
	prefetch_label_.resize(datum_num_label_);
	for (int i = 0; i < datum_num_label_; ++i) {
		(*top)[top_id + i]->Reshape(
			this->layer_param_.batchsize()*delta, 1, 1, 1);
		// LOG(INFO) << "reseting prefetch_label_[top_id + i] " << top_id + i;
		prefetch_label_[i].reset(
			new Blob<Dtype>(this->layer_param_.batchsize()*delta, 1, 1, 1));
	}
	top_id = top_id + datum_num_label_;

	datum_num_weight_ = datum.weight_size();
	// LOG(INFO) << "Setup: datum_num_weight_ " << datum_num_weight_;
	prefetch_weight_.resize(datum_num_weight_);
	for (int i = 0; i < datum_num_weight_; ++i) {
		(*top)[top_id + i]->Reshape(
			this->layer_param_.batchsize()*delta, 1, 1, 1);
		// LOG(INFO) << "reseting prefetch_weight_[top_id + i] " << top_id + i;
		prefetch_weight_[i].reset(
			new Blob<Dtype>(this->layer_param_.batchsize()*delta, 1, 1, 1));
	}
	top_id = top_id + datum_num_weight_;
	CHECK_EQ(top->size(), 4+datum_num_label_+datum_num_weight_-2) << "DataLayerMulti::SetUp needs " 
		<<4+datum_num_label_+datum_num_weight_-2<< "output blobs but gets " << top->size() << " blobs";

	// id
	(*top)[top_id]->Reshape(this->layer_param_.batchsize()*delta, 1, 1, 1);
	prefetch_id_.reset(
			new Blob<int>(this->layer_param_.batchsize()*delta, 1, 1, 1));
	top_id ++;
	
	// datum size
	datum_channels_ = datum.channels();
	datum_height_ = datum.height();
	datum_width_ = datum.width();
	datum_size_ = datum.channels() * datum.height() * datum.width();

	CHECK_GT(datum_height_, cropsize);
	CHECK_GT(datum_width_, cropsize);
	// check if we want to have mean
	if(this->layer_param_.has_meanvalue()){
		LOG(INFO)<<"Loading mean value="<<this->layer_param_.meanvalue();
		int count_=datum.channels()*cropsize*cropsize;
		data_mean_.Reshape(1,datum.channels(),cropsize,cropsize);
		Dtype* data_vec = data_mean_.mutable_cpu_data();
		for(int pos=0;pos<count_;++pos){
			data_vec[pos]=this->layer_param_.meanvalue();
		}
	} else if (this->layer_param_.has_meanfile()) {
		BlobProto blob_proto;
		LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();
		ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(), &blob_proto);
		data_mean_.FromProto(blob_proto);
		CHECK_EQ(data_mean_.num(), 1);
		CHECK_EQ(data_mean_.channels(), datum_channels_);
	//  CHECK_EQ(data_mean_.height(), datum_height_);
	//  CHECK_EQ(data_mean_.width(), datum_width_);
 //   CHECK_EQ(data_mean_.height(), 224);
	//  CHECK_EQ(data_mean_.width(),224);
	} else {
		// Simply initialize an all-empty mean.
		data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
	}
	// Now, start the prefetch thread. Before calling prefetch, we make two
	// cpu_data calls so that the prefetch thread does not accidentally make
	// simultaneous cudaMalloc calls when the main thread is running. In some
	// GPUs this seems to cause failures if we do not so.
	prefetch_data_->mutable_cpu_data();
	for (int i = 0; i < datum_num_label_; ++i) {
		prefetch_label_[i]->mutable_cpu_data();
	}
	for (int i = 0; i < datum_num_weight_; ++i) {
		prefetch_weight_[i]->mutable_cpu_data();
	}
	prefetch_id_->mutable_cpu_data();
	data_mean_.mutable_cpu_data();
	DLOG(INFO) << "Initializing prefetch";
	if(Caffe::phase()==Caffe::TEST){
		CHECK(!pthread_create(&thread_,NULL,DataLayerMultiPrefetchForTest<Dtype>, reinterpret_cast<void*>(this))) << "Pthread execution failed.";
	} else {
		CHECK(!pthread_create(&thread_, NULL, DataLayerMultiPrefetch<Dtype>,reinterpret_cast<void*>(this))) << "Pthread execution failed.";
	}
	DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void DataLayerMulti<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	// First, join the thread
	CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
	int top_id = 0;
	// Copy the data
	memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
			sizeof(Dtype) * prefetch_data_->count());
	top_id ++;

	for (int i = 0; i < datum_num_label_; ++i) {
		memcpy((*top)[top_id + i]->mutable_cpu_data(), prefetch_label_[i]->cpu_data(),
			sizeof(Dtype) * prefetch_label_[i]->count());
	}
	top_id += datum_num_label_;

	for (int i = 0; i < datum_num_weight_; ++i) {
		memcpy((*top)[top_id + i]->mutable_cpu_data(), prefetch_weight_[i]->cpu_data(),
			sizeof(Dtype) * prefetch_weight_[i]->count());
	}
	top_id += datum_num_weight_;

	memcpy((*top)[top_id]->mutable_cpu_data(), prefetch_id_->cpu_data(),
			sizeof(int) * prefetch_id_->count());
	top_id += 1;

	LOG(INFO)<<"data layer forward finished.";
	// Start a new prefetch thread
	if(Caffe::phase()==Caffe::TEST)
	CHECK(!pthread_create(&thread_, NULL, DataLayerMultiPrefetchForTest<Dtype>, reinterpret_cast<void*>(this))) << "Pthread execution failed.";
	else
		CHECK(!pthread_create(&thread_, NULL, DataLayerMultiPrefetch<Dtype>,reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void DataLayerMulti<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	// LOG(INFO)<<"DataLayerMulti::Forward_gpu running";
	// First, join the thread
	CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
	int top_id = 0;
	// Copy the data
	CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
			prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
			cudaMemcpyHostToDevice));
	top_id ++;

	for (int i = 0; i < datum_num_label_; ++i) {
		CUDA_CHECK(cudaMemcpy((*top)[top_id + i]->mutable_gpu_data(),
			prefetch_label_[i]->cpu_data(), sizeof(Dtype) * prefetch_label_[i]->count(),
			cudaMemcpyHostToDevice));
	}
	top_id += datum_num_label_;

	for (int i = 0; i < datum_num_weight_; ++i) {
		CUDA_CHECK(cudaMemcpy((*top)[top_id + i]->mutable_gpu_data(),
			prefetch_weight_[i]->cpu_data(), sizeof(Dtype) * prefetch_weight_[i]->count(),
			cudaMemcpyHostToDevice));
	}
	top_id += datum_num_weight_;

	CUDA_CHECK(cudaMemcpy((*top)[top_id]->mutable_gpu_data(),
			prefetch_id_->cpu_data(), sizeof(int) * prefetch_id_->count(),
			cudaMemcpyHostToDevice));
	top_id ++;

	// Start a new prefetch thread
	if(Caffe::phase()==Caffe::TEST)
		CHECK(!pthread_create(&thread_, NULL, DataLayerMultiPrefetchForTest<Dtype>,reinterpret_cast<void*>(this))) << "Pthread execution failed.";
	else 
		CHECK(!pthread_create(&thread_, NULL, DataLayerMultiPrefetch<Dtype>,reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}


// Save propogate down gradient w.r.t top data.
// The backward operations were dummy - they do not carry any computation.
template <typename Dtype>
Dtype DataLayerMulti<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	// CHECK(Caffe::phase()==Caffe::TEST) << "Calling DataLayerMulti.Backward_cpu in TRAIN";

	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* top_data = top[0]->cpu_data();
	const int* id = reinterpret_cast<const int*>(top[3]->cpu_data());

	const Dtype scale = this->layer_param_.scale();
	const int cropsize = this->layer_param_.cropsize();
	//int mean = this->layer_param_.meanvalue();
	const Dtype* mean = this->data_mean_.cpu_data();
	// Dtype covar_factor = this->layer_param_.covar_factor();
	int channels = this->datum_channels_;
	string dump_path = this->layer_param_.data_dump();
	int num = top[0]->num();
	int dim = top[0]->count() / num;
	
	char filename[256];
	int img_offset = 0;
	int channel_offset = 0;
	Dtype value_in = 0;
	uint8_t value_out = 0;
	Dtype max_val = 0;

	// for Caffe::Test transformed type
	int transformIdx = 0;
	int batchsize = this->layer_param_.batchsize();
	Transforms types;
	if (Caffe::phase()==Caffe::TEST) {
		if (this->layer_param_.has_trans_type()) {
			ReadProtoFromTextFile(this->layer_param_.trans_type(),&types);
			batchsize *= types.transformtype_size();
			LOG(INFO) << "Each image will be processed " << types.transformtype_size() << " times.";
		} else {
			LOG(FATAL) << "Test: No transformation type file.";
		}
		CHECK_EQ(num, batchsize) << "Test batch size do not match";
	}

	for (int itemid = 0; itemid < num; ++itemid ) {
		// data_diff -> image
		if (Caffe::phase()==Caffe::TEST) {
			transformIdx = itemid % types.transformtype_size();
		}

		// Visual Saliency map in color
		// cv::Mat cv_img = cv::Mat::zeros(cropsize,cropsize, CV_8UC3);
		// for (int c = 0; c < channels; ++c) {
		// 	for (int h = 0; h < cropsize; ++h) {
		// 		for (int w = 0; w < cropsize; ++w) {
		// 			img_offset = itemid*channels*cropsize*cropsize;
		// 			channel_offset = c*cropsize*cropsize;

		// 			value_in = top_diff[img_offset + channel_offset + h*cropsize + w ] / covar_factor;
		// 			value_out	= static_cast<uint8_t>( value_in*1.0 / scale + mean[channel_offset + h*cropsize + w] );
		// 			cv_img.at<cv::Vec3b>(h,w)[c] = value_out;
		// 			// if(value_in > 0) {
		// 			// 	LOG(INFO)<<"itemid "<<itemid <<" channel:"<<c <<" height:"<<h <<" width:"<<w <<" value_in:"<<value_in <<" value_out:"<<value_out;
		// 			// }
		// 		}
		// 	}
		// }
		// // save image files
		// sprintf( filename, "%s/saliency/%d.png", dump_path.c_str(), itemid );
		// // sprintf( filename, "%s/saliency/%d_%d.png", dump_path.c_str(), id[itemid], transformIdx );
		// cv::imwrite( filename, cv_img );

		std::ofstream outfile;
		// sprintf( filename, "%s/saliency/%d.txt", dump_path.c_str(), itemid );
		sprintf( filename, "%s/saliency/%d_%d.txt", dump_path.c_str(), id[itemid], transformIdx );
		outfile.open(filename);
		for (int h = 0; h < cropsize; ++h) {
			for (int w = 0; w < cropsize; ++w) {
				max_val = 0;
				for (int c = 0; c < channels; ++c) {
					img_offset = itemid*channels*cropsize*cropsize;
					channel_offset = c*cropsize*cropsize;
					value_in = top_diff[img_offset + channel_offset + h*cropsize + w ] / scale;
					if (abs(value_in) > max_val) {
						max_val = abs(value_in);
					}
					// if(value_in > 0) {
					// 	LOG(INFO)<<"itemid "<<itemid <<" channel:"<<c <<" height:"<<h <<" width:"<<w <<" value_in:"<<value_in <<" value_out:"<<value_out;
					// }
				}
				outfile << max_val << "\t";
			}
			outfile << std::endl;
		}
		// save image files
		outfile.close();


		// data_diff -> image
		cv::Mat original_img = cv::Mat::zeros(cropsize,cropsize, CV_8UC3);
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < cropsize; ++h) {
				for (int w = 0; w < cropsize; ++w) {
					img_offset = itemid*channels*cropsize*cropsize;
					channel_offset = c*cropsize*cropsize;

					value_in = top_data[img_offset + channel_offset + h*cropsize + w ];
					value_out	= static_cast<uint8_t>( value_in*1.0 / scale + mean[channel_offset + h*cropsize + w] );
					original_img.at<cv::Vec3b>(h,w)[c] = value_out;
					//if(value_in > 0) {
						//LOG(INFO)<<"itemid "<<itemid <<" channel:"<<c <<" height:"<<h <<" width:"<<w <<" value_in:"<<value_in <<" value_out:"<<value_out;
					//}
				}
			}
		}
		// save image files
		// sprintf( filename, "%s/original/%d.png", dump_path.c_str(), itemid );
		sprintf( filename, "%s/original/%d_%d.png", dump_path.c_str(), id[itemid], transformIdx );
		cv::imwrite( filename, original_img );

	}

	return Dtype(0.);
}

template <typename Dtype>
Dtype DataLayerMulti<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	// using cpu version
	return Backward_cpu(top, propagate_down, bottom);
	//return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayerMulti);

}  // namespace caffe
