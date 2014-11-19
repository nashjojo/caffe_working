// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;

namespace caffe {

void ReadProtoFromTextFile(const char* filename,
		::google::protobuf::Message* proto) {
	int fd = open(filename, O_RDONLY);
	CHECK_NE(fd, -1) << "File not found: " << filename;
	FileInputStream* input = new FileInputStream(fd);
	CHECK(google::protobuf::TextFormat::Parse(input, proto));
	delete input;
	close(fd);
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
	int fd = open(filename, O_WRONLY);
	FileOutputStream* output = new FileOutputStream(fd);
	CHECK(google::protobuf::TextFormat::Print(proto, output));
	delete output;
	close(fd);
}

void ReadProtoFromBinaryFile(const char* filename, Message* proto) {
	int fd = open(filename, O_RDONLY);
	CHECK_NE(fd, -1) << "File not found: " << filename;
	ZeroCopyInputStream* raw_input = new FileInputStream(fd);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	// coded_input->SetTotalBytesLimit(536870912, 268435456);	 //limit the largest number of protobuf
	coded_input->SetTotalBytesLimit(536870912*10, 268435456*10);

	CHECK(proto->ParseFromCodedStream(coded_input));

	delete coded_input;
	delete raw_input;
	close(fd);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
	fstream output(filename, ios::out | ios::trunc | ios::binary);
	CHECK(proto.SerializeToOstream(&output));
}

bool ReadImageToDatum(const string& filename, const int label,
		const int height, const int width, Datum* datum) {
	cv::Mat cv_img;
	if (height > 0 && width > 0) {
		cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
		cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else {
		cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	}
	if (!cv_img.data) {
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	if (height > 0 && width > 0) {

	}
	datum->set_channels(3);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
			}
		}
	}
	return true;
}
// Kaixiang MO, 23th April, 2014
// Reading for DatumWeighted
bool ReadImageToDatumWeighted(const string& filename, const int label, const float weight, const long int id, 
		const int height, const int width, DatumWeighted* datum) {
	cv::Mat cv_img;
	if (height > 0 && width > 0) {
		cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
		cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else {
		cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	}
	if (!cv_img.data) {
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	if (height > 0 && width > 0) {

	}
	datum->set_channels(3);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->set_label(label);
	datum->set_weight(weight);
	datum->set_id(id);
	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
			}
		}
	}
	return true;
}
// Resize the smaller edge to a fix point, then insert to leveldb.
bool ReadResizedImageToDatumWeighted(const string& filename, const int label, const float weight, const long int id, 
		const int short_edge, DatumWeighted* datum) {
	cv::Mat cv_img;
	cv::Mat cv_img_origin;
	int height;
	int width;
	cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	//CHECK_EQ(cv_img_origin.channels(), 3) << "Image need to have exactly 3 channels.";
	if (cv_img_origin.channels() != 3) {
		LOG(ERROR) << "Image need to have exactly 3 channels.\t" << filename;
		return false;
	}
	height = cv_img_origin.size().height;
	width = cv_img_origin.size().width;
	// what if both edge is shorter?
	// find the shorted edge
	if (height<width && height<short_edge) {
	float ratio = (1.0*short_edge)/height;
	height = short_edge;
	width = int(ratio*width);
		cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else if (width<height && width<short_edge) {
	float ratio = (1.0*short_edge)/width;
	width = short_edge;
	height = int(ratio*height);
	cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else {
	cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	}
	if (!cv_img.data) {
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	datum->set_channels(3);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->set_label(label);
	datum->set_weight(weight);
	datum->set_id(id);
	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
			}
		}
	}
	return true;
}

// Resize the smaller edge to a fix point, then insert to leveldb. With extra features as well
bool ReadResizedImageToDatumWeighted(const string& filename, const int label, const float weight, const long int id, 
		const int short_edge, DatumWeighted* datum, const std::vector<float>& feature) {
	cv::Mat cv_img;
	cv::Mat cv_img_origin;
	int height;
	int width;
	cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	//CHECK_EQ(cv_img_origin.channels(), 3) << "Image need to have exactly 3 channels.";
	if (cv_img_origin.channels() != 3) {
		LOG(ERROR) << "Image need to have exactly 3 channels.\t" << filename;
		return false;
	}
	height = cv_img_origin.size().height;
	width = cv_img_origin.size().width;
	// what if both edge is shorter?
	// find the shorted edge
	if (height<width && height<short_edge) {
	float ratio = (1.0*short_edge)/height;
	height = short_edge;
	width = int(ratio*width);
		cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else if (width<height && width<short_edge) {
	float ratio = (1.0*short_edge)/width;
	width = short_edge;
	height = int(ratio*height);
	cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else {
	cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	}
	if (!cv_img.data) {
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	datum->set_channels(3);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->set_label(label);
	datum->set_weight(weight);
	datum->set_id(id);
	
	// setting extra features
	datum->clear_extfeature();
	for (int i = 0; i < feature.size(); i++) {
		datum->add_extfeature(feature[i]);
	}
	
	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
			}
		}
	}
	return true;
}
// ~Kaixiang MO, 23th April, 2014

// Kaixiang Mo, 28th June, 2014
// have postive and negative weight, dummy label
// Resize the smaller edge to a fix point, then insert to leveldb. With extra features as well
bool ReadResizedImageToDatumPosNeg(const string& filename, const float weight, const float neg_weight, const long int id,
		const int short_edge, DatumPosNeg* datum, const std::vector<float>& feature) {
	cv::Mat cv_img;
	cv::Mat cv_img_origin;
	int height;
	int width;
	cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	//CHECK_EQ(cv_img_origin.channels(), 3) << "Image need to have exactly 3 channels.";
	if (cv_img_origin.channels() != 3) {
		LOG(ERROR) << "Image need to have exactly 3 channels.\t" << filename;
		return false;
	}
	height = cv_img_origin.size().height;
	width = cv_img_origin.size().width;
	// what if both edge is shorter?
	// find the shorted edge
	if (height<width && height<short_edge) {
	float ratio = (1.0*short_edge)/height;
	height = short_edge;
	width = int(ratio*width);
		cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else if (width<height && width<short_edge) {
	float ratio = (1.0*short_edge)/width;
	width = short_edge;
	height = int(ratio*height);
	cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else {
	cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	}
	if (!cv_img.data) {
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	datum->set_channels(3);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->set_label(0);
	datum->set_weight(weight);
	datum->set_neg_weight(neg_weight);
	datum->set_id(id);
	
	// setting extra features
	datum->clear_extfeature();
	for (int i = 0; i < feature.size(); i++) {
		datum->add_extfeature(feature[i]);
	}
	
	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
			}
		}
	}
	return true;
} // ~ Kaixiang MO, 28th June, 2014

}  // namespace caffe
