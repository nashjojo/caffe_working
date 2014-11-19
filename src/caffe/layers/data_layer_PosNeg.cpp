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

// Kaixiang MO, 21 April, 2014
#include <iostream>

//#define sideLen 32
using namespace std;
using std::string;

namespace caffe {

template <typename Dtype>
void* DataLayerPosNegPrefetchForTest(void* layer_pointer) {
	CHECK(layer_pointer);
	DataLayerPosNeg<Dtype>* layer = reinterpret_cast<DataLayerPosNeg<Dtype>*>(layer_pointer);
	CHECK(layer);
	DatumPosNeg datum;
	CHECK(layer->prefetch_data_);
	Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
	Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
	Dtype* top_weight = layer->prefetch_weight_->mutable_cpu_data();
	Dtype* top_neg_weight = layer->prefetch_neg_weight_->mutable_cpu_data();
	Dtype* top_extfeature = layer->prefetch_extfeature_->mutable_cpu_data();
	int* top_id = layer->prefetch_id_->mutable_cpu_data();
	const Dtype scale = layer->layer_param_.scale();
	int batchsize = layer->layer_param_.batchsize();
	const int cropsize = layer->layer_param_.cropsize();
	const bool mirror = layer->layer_param_.mirror();
	const int num_extfeature = layer->layer_param_.num_extfeature();
	CHECK(num_extfeature > 0) << "Number of extra feature should be larger than 0.";

	if (mirror && cropsize == 0) {
		LOG(FATAL) << "Current implementation requires mirror and cropsize to be "
				<< "set at the same time.";
	}
	ofstream out;
	if(layer->layer_param_.has_test_log()){
		out.open(layer->layer_param_.test_log().c_str(),ios::out|ios::app);
	}
	// datum scales
	const int channels = layer->datum_channels_;
	//const int height = layer->datum_height_;
	//const int width = layer->datum_width_;
	const int size = layer->datum_size_; // number of instance
	const Dtype* mean = layer->data_mean_.cpu_data();
	Transforms types;
	if(layer->layer_param_.has_trans_type()){
		ReadProtoFromTextFile(layer->layer_param_.trans_type(),&types);
		batchsize*=types.transformtype_size();
	 // LOG(INFO)<<"batchsize="<<batchsize;
	} else if(layer->layer_param_.has_trans_type_default()){
		ReadProtoFromTextFile(layer->layer_param_.trans_type_default(),&types);
	} else {
		LOG(FATAL)<<"No transformation type file.";
	}
	int sideLen=types.side_len();
	int height,width;
	
	for (int itemid = 0; itemid < batchsize; itemid++){
		// get a blob
		if(itemid%types.transformtype_size()==0){
			CHECK(layer->iter_);
			CHECK(layer->iter_->Valid());
			datum.ParseFromString(layer->iter_->value().ToString());
		}

//    LOG(INFO)<<"itemid="<<itemid;
		const string& data = datum.data();
		height = datum.height();
		width = datum.width();
		if (cropsize) {
			CHECK(data.size()) << "Image cropping only support uint8 data";
			int h_off, w_off;
			const TransformParameter& transParam=types.transformtype(
				itemid%types.transformtype_size());
			int t_size=transParam.size();
			string t_pos=transParam.pos(),t_side=transParam.side();
			bool t_mirror=transParam.mirror();
			cv::Mat beforeTrans=cv::Mat::zeros(height,width,CV_8UC3);
			cv::Mat afterCut,afterCrop,afterMirror,afterResize;
			for(int c=0;c<channels;++c){
				for(int h=0;h<beforeTrans.rows;++h){
					for(int w=0;w<beforeTrans.cols;++w){
						beforeTrans.at<cv::Vec3b>(h,w)[c]=data[(c*height+h)*width+w];
					}
				}
			}
	//    LOG(INFO)<<"before cut right.";
			afterCut=cv::Mat(cv::Size(sideLen,sideLen),CV_8UC3);
	 //   LOG(INFO)<<"after cur right.";
			caffe::GetWidthAndHeightOff(true, t_side, height, width, sideLen, h_off, w_off);
		//  LOG(INFO)<<"get off right.";
			cv::Rect myROI(w_off,h_off, sideLen, sideLen);
		//  LOG(INFO)<<"myroi right.";
			afterCut=beforeTrans(myROI);
		 // LOG(INFO)<<"after cur myroi right.";
			afterCrop=cv::Mat(cv::Size(t_size,t_size),CV_8UC3);
		 // LOG(INFO)<<"after crop right.";
			if(t_size<sideLen){
				caffe::GetWidthAndHeightOff(false, t_pos, sideLen, sideLen, t_size, h_off, w_off);
			 // LOG(INFO)<<sideLen<<" "<<w_off<<" "<<h_off<<" "<<t_size;
				afterCrop=afterCut(cv::Rect(w_off,h_off,t_size,t_size));
			} else {
				afterCrop=afterCut.clone();
			}
			afterMirror=cv::Mat(t_size,t_size,CV_8UC3);
			if(t_mirror){
				cv::flip(afterCrop,afterMirror,1); // the last parameter should be 1 to flip lr 
			} else {
				afterMirror=afterCrop.clone();
			}
			afterResize = cv::Mat(cropsize,cropsize, CV_8UC3);  
			cv::resize(afterMirror,afterResize,cv::Size(cropsize,cropsize),0,0,CV_INTER_CUBIC);
			// Normal copy
			for (int c = 0; c < channels; ++c) {
				for (int h = 0; h < cropsize; ++h) {
					for (int w = 0; w < cropsize; ++w) {
						top_data[((itemid * channels + c) * cropsize + h) * cropsize + w]
								= (static_cast<Dtype>((uint8_t)afterResize.at<cv::Vec3b>(h,w)[c])
									- mean[(c * cropsize + h ) * cropsize + w]) * scale;
					}
				}
			}
			//top_label[itemid]=datum.label();
		//top_weight[itemid] = datum.weight();
		//top_id[itemid] = datum.id();
		
			if(layer->layer_param_.has_test_log()){
					out<<itemid<<" "<<top_label[itemid]<<" weight"<<top_weight[itemid]<<" neg_weight"<<top_neg_weight[itemid]<<" id"<<top_id[itemid]<<endl;
			}
		} else {
			// we will prefer to use data() first, and then try float_data()
			if (data.size()) {
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
			//top_label[itemid]=datum.label();
		//top_weight[itemid] = datum.weight();
		//top_id[itemid] = datum.id();
		}
	// copy other fields
	top_label[itemid] = datum.label();
	top_weight[itemid] = datum.weight();
	top_neg_weight[itemid] = datum.neg_weight();
	top_id[itemid] = datum.id();
	// have to use a for loop to assign
	// std::cout << "DataLayerPosNegPrefetchForTest" << std::endl;
	for (int i = 0; i < num_extfeature; i++) {
		// If we consider only category information, we set num_extfeature to 5
		// We hard code in this code that there are total 10 num_extfeatures
		top_extfeature[itemid * num_extfeature + i] = static_cast<Dtype>(datum.extfeature(i));
		// std::cout << datum.extfeature(i) <<"\t";
	}
	// std::cout << std::endl;
	
		// go to the next iter
		if(itemid%types.transformtype_size()+1==types.transformtype_size()){
			layer->iter_->Next();
			if (!layer->iter_->Valid()) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				layer->iter_->SeekToFirst();
			}
		}
//	LOG(INFO)<<"finish transform for itemid "<<itemid;
	}

	if(layer->layer_param_.has_test_log()){
		out.close();
	}
	LOG(INFO)<<"setup complete.";
	return (void*)NULL;
}

template <typename Dtype>
void* DataLayerPosNegPrefetch(void* layer_pointer) {
	CHECK(layer_pointer);
	DataLayerPosNeg<Dtype>* layer = reinterpret_cast<DataLayerPosNeg<Dtype>*>(layer_pointer);
	CHECK(layer);
	DatumPosNeg datum;
	CHECK(layer->prefetch_data_);
	Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
	Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
	Dtype* top_weight = layer->prefetch_weight_->mutable_cpu_data();
	Dtype* top_neg_weight = layer->prefetch_neg_weight_->mutable_cpu_data();
	Dtype* top_extfeature = layer->prefetch_extfeature_->mutable_cpu_data();
	int* top_id = layer->prefetch_id_->mutable_cpu_data();
	const Dtype scale = layer->layer_param_.scale();
	const int batchsize = layer->layer_param_.batchsize();
	const int cropsize = layer->layer_param_.cropsize();
	const bool mirror = layer->layer_param_.mirror();
	const float luminance_vary = layer->layer_param_.luminance_vary();
	const float contrast_vary = layer->layer_param_.contrast_vary();
	const int num_extfeature = layer->layer_param_.num_extfeature();
	CHECK(num_extfeature > 0) << "Number of extra feature should be larger than 0.";
	
	if (mirror && cropsize == 0) {
		LOG(FATAL) << "Current implementation requires mirror and cropsize to be "
				<< "set at the same time.";
	}
	// datum scales
	const int channels = layer->datum_channels_;
	const int size = layer->datum_size_;
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
				if(layer->layer_param_.has_resolve_size()){
					caffe::GetOffAndResolvesize(height,width,resolves,h_off,w_off,resolvesize);
				} else {
					h_off = rand() % (height - resolvesize + 1);
					w_off = rand() % (width - resolvesize + 1);
				}
			} else {
			 // resolvesize=cropsize;
				h_off = (datum.height() - resolvesize) / 2;
				w_off = (datum.width() - resolvesize) / 2;  
			}
			beforeResize=cv::Mat::zeros(resolvesize,resolvesize,CV_8UC3);
			for(int c=0;c<channels;++c){
				for(int h=0;h<beforeResize.rows;++h){
					for(int w=0;w<beforeResize.cols;++w){
						beforeResize.at<cv::Vec3b>(h,w)[c]=data[(c*height+h+h_off)*width+w+w_off];
					}
				}
			}
			afterResize = cv::Mat(cropsize,cropsize, CV_8UC3); 
			cv::resize(beforeResize,afterResize,cv::Size(cropsize,cropsize),0,0,CV_INTER_CUBIC);
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
		top_label[itemid] = datum.label();
	top_weight[itemid] = datum.weight();
	top_neg_weight[itemid] = datum.neg_weight();
	top_id[itemid] = datum.id();
	// have to use a for loop to assign
	for (int i = 0; i < num_extfeature; i++) {
		// If we consider only category information, we set num_extfeature to 5
		// We hard code in this code that there are total 10 num_extfeatures
		top_extfeature[itemid * num_extfeature + i] = static_cast<Dtype>(datum.extfeature(i));
	}
	// check the weight here
	//std::cout << "item " << itemid << " weight " << datum.weight() <<" id " << datum.id() << std::endl;
	
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
DataLayerPosNeg<Dtype>::~DataLayerPosNeg<Dtype>() {
	// Finally, join the thread
	CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
void DataLayerPosNeg<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	LOG(INFO)<<"DataLayerPosNeg::SetUp running";
	CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
	// 3rd is weight blob, 4rd is id blob, 5th blob is extra feature.
	CHECK_EQ(top->size(), 6) << "Data Layer takes 6 blobs as output, the 3:weight, 4:instance-id 5:extra-features, 6:negative-weight";
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
	leveldb::Iterator* itr;
	iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
	iter_->SeekToFirst();
	LOG(INFO)<<"seek first complete";
	// Check if we would need to randomly skip a few data points
	if (this->layer_param_.rand_skip()) {
		unsigned int skip = rand() % this->layer_param_.rand_skip();
		LOG(INFO) << "Skipping first " << skip << " data points.";
		while (skip-- > 0) {
			iter_->Next();
			if (!iter_->Valid()) {
				iter_->SeekToFirst();
			}
		}
	}
	// Read a data point, and use it to initialize the top blob.
	DatumPosNeg datum;
	datum.ParseFromString(iter_->value().ToString());
	// image
	LOG(INFO)<<"parse first complete";
	int cropsize = this->layer_param_.cropsize();
	Transforms types;
	int delta=1;
	if(this->layer_param_.has_trans_type()){
		ReadProtoFromTextFile(this->layer_param_.trans_type(),&types);
		delta*=types.transformtype_size();
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
	// label
	(*top)[1]->Reshape(this->layer_param_.batchsize()*delta, 1, 1, 1);
	prefetch_label_.reset(
			new Blob<Dtype>(this->layer_param_.batchsize()*delta, 1, 1, 1));
	// weight
	(*top)[2]->Reshape(this->layer_param_.batchsize()*delta, 1, 1, 1);
	prefetch_weight_.reset(
			new Blob<Dtype>(this->layer_param_.batchsize()*delta, 1, 1, 1));
		// negative weight
	(*top)[5]->Reshape(this->layer_param_.batchsize()*delta, 1, 1, 1);
	prefetch_neg_weight_.reset(
			new Blob<Dtype>(this->layer_param_.batchsize()*delta, 1, 1, 1));

	// id
	(*top)[3]->Reshape(this->layer_param_.batchsize()*delta, 1, 1, 1);
	prefetch_id_.reset(
			new Blob<int>(this->layer_param_.batchsize()*delta, 1, 1, 1));

	const int num_extfeature = this->layer_param_.num_extfeature();
	CHECK(num_extfeature > 0) << "Number of extra feature should be larger than 0.";
	// external features
	(*top)[4]->Reshape(
			this->layer_param_.batchsize()*delta, num_extfeature, 1, 1);
	prefetch_extfeature_.reset(new Blob<Dtype>(
			this->layer_param_.batchsize()*delta, num_extfeature, 1, 1));
	
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
	prefetch_label_->mutable_cpu_data();
	prefetch_weight_->mutable_cpu_data();
	prefetch_neg_weight_->mutable_cpu_data();
	prefetch_id_->mutable_cpu_data();
	data_mean_.cpu_data();
	DLOG(INFO) << "Initializing prefetch";
	if(Caffe::phase()==Caffe::TEST){
		CHECK(!pthread_create(&thread_,NULL,DataLayerPosNegPrefetchForTest<Dtype>, reinterpret_cast<void*>(this))) << "Pthread execution failed.";
	} else {
		CHECK(!pthread_create(&thread_, NULL, DataLayerPosNegPrefetch<Dtype>,reinterpret_cast<void*>(this))) << "Pthread execution failed.";
	}
	DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void DataLayerPosNeg<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	// First, join the thread
	CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
	// Copy the data
	memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
			sizeof(Dtype) * prefetch_data_->count());
	memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
			sizeof(Dtype) * prefetch_label_->count());
	memcpy((*top)[2]->mutable_cpu_data(), prefetch_weight_->cpu_data(),
			sizeof(Dtype) * prefetch_weight_->count());
	memcpy((*top)[5]->mutable_cpu_data(), prefetch_neg_weight_->cpu_data(),
			sizeof(Dtype) * prefetch_neg_weight_->count());
	memcpy((*top)[3]->mutable_cpu_data(), prefetch_id_->cpu_data(),
			sizeof(int) * prefetch_id_->count());
	memcpy((*top)[4]->mutable_cpu_data(), prefetch_extfeature_->cpu_data(),
			sizeof(Dtype) * prefetch_extfeature_->count());
	LOG(INFO)<<"data layer forward finished.";
	// Start a new prefetch thread
	if(Caffe::phase()==Caffe::TEST)
	CHECK(!pthread_create(&thread_, NULL, DataLayerPosNegPrefetchForTest<Dtype>, reinterpret_cast<void*>(this))) << "Pthread execution failed.";
	else
		CHECK(!pthread_create(&thread_, NULL, DataLayerPosNegPrefetch<Dtype>,reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void DataLayerPosNeg<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
	// LOG(INFO)<<"DataLayerPosNeg::Forward_gpu running";
	// First, join the thread
	CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
	// Copy the data
	CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
			prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
			cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
			prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
			cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy((*top)[2]->mutable_gpu_data(),
			prefetch_weight_->cpu_data(), sizeof(Dtype) * prefetch_weight_->count(),
			cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy((*top)[5]->mutable_gpu_data(),
			prefetch_neg_weight_->cpu_data(), sizeof(Dtype) * prefetch_neg_weight_->count(),
			cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy((*top)[3]->mutable_gpu_data(),
			prefetch_id_->cpu_data(), sizeof(int) * prefetch_id_->count(),
			cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy((*top)[4]->mutable_gpu_data(),
			prefetch_extfeature_->cpu_data(), sizeof(Dtype) * prefetch_extfeature_->count(),
			cudaMemcpyHostToDevice));
	// Start a new prefetch thread
	if(Caffe::phase()==Caffe::TEST)
		CHECK(!pthread_create(&thread_, NULL, DataLayerPosNegPrefetchForTest<Dtype>,reinterpret_cast<void*>(this))) << "Pthread execution failed.";
	else 
		CHECK(!pthread_create(&thread_, NULL, DataLayerPosNegPrefetch<Dtype>,reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype DataLayerPosNeg<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	return Dtype(0.);
}

template <typename Dtype>
Dtype DataLayerPosNeg<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayerPosNeg);

}  // namespace caffe
