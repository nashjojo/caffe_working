// Copyright 2013 Yangqing Jia
// This program converts a set of images to a leveldb by storing them as DatumPosNeg
// proto buffers.
// Usage:
// 		0 						1 					2 				3 			4 				
//    convert_multi ROOTFOLDER/ LISTFILE 	DB_NAME shuffle 
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   urls			adid        <label, weight>
//   201.jpg	20013123    
//   ....
// We will shuffle the file if the last digit is 1. 
#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <typeinfo>

using namespace caffe;
using std::vector;
using std::string;

const int NUM_LABEL=3;
const int MIN_IMG_SIZE=160;
const int MAX_IMG_SIZE=192;
const int WRITE_BATCH_SIZE = 20000;

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc < 5) {
		printf("Convert a set of images to the leveldb format used\n"
				"as input for Caffe.\n"
				"Usage:\n"
				"    convert_multi ROOTFOLDER/ LISTFILE DB_NAME"
				" RANDOM_SHUFFLE_DATA[0 or 1]\n");
		return 0;
	}
	std::ifstream infile(argv[2]);
	std::vector< string > imgUrls;
	std::vector< int > adid_set; // store each line
	std::vector< std::vector<int> > labels_set;
	std::vector< std::vector<double> > weights_set;
	std::vector< int> order;
	//string filename;
	string urls;
	int adid;
	int label[NUM_LABEL];
	int weight[NUM_LABEL];
	string line;
	int line_count = 0;
	
	// reading all labels please
	while (getline(infile,line)) {
		std::vector<int> temp_labels;
		std::vector<double> temp_weights;

		// split the string
		std::istringstream lineString(line);
		lineString >> urls >> adid;
		
		for (int i = 0; i < NUM_LABEL; i++) {
			lineString >> label[i] >> weight[i];
			temp_labels.push_back(label[i]);
			temp_weights.push_back(weight[i]);
		}

		adid_set.push_back(adid);
		labels_set.push_back(temp_labels);
		weights_set.push_back(temp_weights);
		imgUrls.push_back(urls);
		order.push_back(line_count);
		line_count ++;
	}

	/* Cannot shuffle, because of external imgUrls array*/
	if (string(argv[4]) == "shuffle") {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		std::random_shuffle(order.begin(), order.end());
	}
	LOG(INFO) << "A total of " << adid_set.size() << " images.";

	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 1024*1024*100; // 100 MB
	options.max_open_files = 100;
	options.block_size = 1024*1024*100; // 100 MB
	leveldb::ReadOptions read_options;
	read_options.fill_cache = false;
	LOG(INFO) << "Opening leveldb " << argv[3];
	leveldb::Status status = leveldb::DB::Open(
			options, argv[3], &db);
	CHECK(status.ok()) << "Failed to open leveldb " << argv[3];
	LOG(INFO) << "Leveldb checked finished";

	string root_folder(argv[1]);
	DatumMulti datum;
	int count = 0;
	char key[256];
	char filename[256];
	leveldb::WriteBatch* batch = new leveldb::WriteBatch();
	for (int line_id = 0; line_id < order.size(); ++line_id) {
		// LOG(INFO) << "line_id " << adid_set[ order[line_id] ];
		sprintf( filename, "%s/%s.jpg", root_folder.c_str(), imgUrls[ order[line_id] ].c_str() );

		datum.clear_data();
		datum.clear_float_data();
		string* datum_string = datum.mutable_data();
		int rows, cols;
		if (!ReadResizedImage(string(filename), 
			MIN_IMG_SIZE, MAX_IMG_SIZE, datum_string, rows, cols)) {
			continue;
		};
		datum.set_channels(3);
		datum.set_height(rows);
		datum.set_width(cols);

		// set label
		datum.clear_label();
		for (int i = 0; i < NUM_LABEL; i++) {
			datum.add_label( labels_set[ order[line_id] ][i] );
		}
		// set weight
		datum.clear_weight();
		for (int i = 0; i < NUM_LABEL; i++) {
			datum.add_weight( weights_set[ order[line_id] ][i] );
		}
		// set id
		datum.set_id(adid_set[ order[line_id] ]);

		sprintf(key, "%d", adid_set[ order[line_id] ]);
		string value; 
		
		datum.SerializeToString(&value);
		batch->Put(string(key), value);
		count ++;
		if (count % WRITE_BATCH_SIZE == 0) {
			db->Write(leveldb::WriteOptions(), batch);
			LOG(INFO) << "Processed " << count << " files.";
			delete batch;
			batch = new leveldb::WriteBatch();
		}
	}
	// write the last batch
	if (count % WRITE_BATCH_SIZE != 0) {
		db->Write(leveldb::WriteOptions(), batch);
		LOG(INFO) << "Processed " << count << " files.";
	}

	delete batch;
	delete db;
	return 0;
}
