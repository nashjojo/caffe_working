// Copyright Yangqing Jia 2013

#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <google/protobuf/message.h>

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;
using ::google::protobuf::Message;

namespace caffe {

void ReadProtoFromTextFile(const char* filename,
		Message* proto);
inline void ReadProtoFromTextFile(const string& filename,
		Message* proto) {
	ReadProtoFromTextFile(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
	WriteProtoToTextFile(proto, filename.c_str());
}

void ReadProtoFromBinaryFile(const char* filename,
		Message* proto);
inline void ReadProtoFromBinaryFile(const string& filename,
		Message* proto) {
	ReadProtoFromBinaryFile(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
		const Message& proto, const string& filename) {
	WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadImageToDatum(const string& filename, const int label,
		const int height, const int width, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
		Datum* datum) {
	return ReadImageToDatum(filename, label, 0, 0, datum);
}

// Kaixiang MO, 23th April, 2014
bool ReadImageToDatumWeighted(const string& filename, const int label, const float weight, const int id,
		const int height, const int width, DatumWeighted* datum);

inline bool ReadImageToDatumWeighted(const string& filename, const int label, const float weight, const int id,
		DatumWeighted* datum) {
	return ReadImageToDatumWeighted(filename, label, weight, id, 0, 0, datum);
}

bool ReadResizedImageToDatumWeighted(const string& filename, const int label, const float weight, const int id,
		const int short_edge, DatumWeighted* datum);
	
// with extra feature
bool ReadResizedImageToDatumWeighted(const string& filename, const int label, const float weight, const int id,
		const int short_edge, DatumWeighted* datum, const std::vector<float>& feature);
// ~Kaixiang MO, 23th April, 2014

// pos and neg weight in one instance, with extra feature
bool ReadResizedImageToDatumPosNeg(const string& filename, const float weight, const float neg_weight, const int id,
		const int short_edge, DatumPosNeg* datum, const std::vector<float>& feature);
// ~Kaixiang MO, 28th June, 2014

// Resize long side to a fixed length
bool ReadResizedLargeImageToDatumPosNeg(const string& filename, const float weight, const float neg_weight, const int id,
		const int short_edge, const int long_edge, DatumPosNeg* datum, const std::vector<float>& feature);
// ~Kaixiang MO, 24th Nov, 2014

// Resize img to DatumMulti
bool ReadResizedImage(const string& filename, int short_edge, int long_edge, string* datum_string, int& rows, int& cols);

bool ReadResizedImageToDatumMulti(const string& filename, int short_edge, int long_edge, DatumMulti* datum);
}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
