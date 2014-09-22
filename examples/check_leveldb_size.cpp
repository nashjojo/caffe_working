// Copyright 2013 Yangqing Jia
// This program converts a set of images to a leveldb by storing them as DatumPosNeg
// proto buffers.
// Usage:
//    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   adid        #click  #impression
//   20013123    1013    1021312
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

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc < 2) {
		printf("Check leveldb number of records.\n"
				"Usage:\n"
				"    check_leveldb_size DB_NAME\n");
		return 0;
	}

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;

	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = false;
	options.create_if_missing = false;
	// leveldb::ReadOptions read_options;
	// read_options.fill_cache = false;
	LOG(INFO) << "Opening leveldb " << argv[1];
	leveldb::Status status = leveldb::DB::Open(
			options, argv[1], &db);
	CHECK(status.ok()) << "Failed to open leveldb " << argv[1];
	LOG(INFO) << "Leveldb checked finished";
	db_.reset(db);

	leveldb::Iterator* itr;
	iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
	iter_->SeekToFirst();

	int count = 0;
	while(iter_->Valid()) {
		iter_->Next();
		count ++;
		if (count%1000==0) {
			std::cout << "checked " << count << " records." << std::endl;
		}
	}

	std::cout << argv[1] << " has " << count << " records." << std::endl;

	// delete db;
	return 0;
}
