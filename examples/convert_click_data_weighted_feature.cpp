// Copyright 2013 Yangqing Jia
// This program converts a set of images to a leveldb by storing them as DatumWeighted
// proto buffers.
// Usage:
//    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   20013123 1 1013
//   20123123 0 1021312
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

const int IMG_SIZE=128;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 4) {
    printf("Convert a set of images to the leveldb format used\n"
        "as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME"
        " RANDOM_SHUFFLE_DATA[0 or 1]\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
    return 0;
  }
  std::ifstream infile(argv[2]);
  std::vector< std::vector<int> > lines; // store each line
  std::vector< std::vector<float> > features; // store features of each line
  //string filename;
  int adid;
  int label;
  int weight;
  string line;
  float feature;
  int feature_id;
  
  // while (infile >> adid >> label >> weight) {
    // std::cout << adid << label << weight << std::endl;
    // int temp[] = {adid,label,weight};
	// std::vector<int> temp_vector(temp, temp + sizeof(temp)/sizeof(int));
    // lines.push_back(temp_vector);
    // std::cout << temp_vector[0] << temp_vector[1] << temp_vector[2] << std::endl;
  // }
  
  while (getline(infile,line)) {
	// split the string
	std::istringstream lineString(line);
	lineString >> adid >> label >> weight;
	// std::cout << "reading from file" << std::endl;
	// std::cout << adid <<" "<< label <<" "<< weight <<" "<< std::endl;
	int temp[] = {adid,label,weight};
	std::vector<int> temp_vector(temp, temp + sizeof(temp)/sizeof(int));
	lines.push_back(temp_vector);
	
	// read in extra features
	std::vector<float> temp_feature;
	feature_id = 0;
	while ( lineString >> feature ) {
	  temp_feature.push_back(feature);
	  feature_id ++;
	  // std::cout << feature_id <<" "<< feature << std::endl;
	}
	// std::cout << "total feature " << feature_id << std::endl;
	
	// std::cout << "reading from file" << std::endl;
	// std::cout << temp_feature.size() << std::endl;
	// for (int i = 0; i < temp_feature.size(); i++ ) {
		// std::cout << temp_feature[i] << " ";
	// }
	// std::cout << std::endl;
	features.push_back(temp_feature);
  }
  
  if (argc == 5 && argv[4][0] == '1') {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    std::random_shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

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
  DatumWeighted datum;
  int count = 0;
  char key[256];
  char filename[256];
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
	//LOG(INFO) << "line_id" << line_id;
	//LOG(INFO) << lines[line_id][0] << lines[line_id][1] << lines[line_id][2];
	sprintf( filename, "%s%d.jpg", root_folder.c_str(), lines[line_id][0]);
	//LOG(INFO) << filename <<" "<< lines[line_id][1] <<" "<< lines[line_id][2];

    if (!ReadResizedImageToDatumWeighted(string(filename), 
			lines[line_id][1], float(lines[line_id][2]), lines[line_id][0], 
			IMG_SIZE, &datum, features[line_id])) {
      continue;
    };
    // sequential
    //key = ""+lines[line_id][0]+"_"+lines[line_id][1];
	sprintf(key, "%d_%d", lines[line_id][0], lines[line_id][1]);
	//LOG(INFO) << key <<" ";
    string value; 

	// std::cout << "reading from datum" << std::endl;
	// for (int i = 0; i < datum.extfeature_size(); i++ ) {
	  // std::cout << datum.extfeature(i) << " ";
	// }
	// std::cout << std::endl;
    // get the value
    //std::cout << typeid(datum.id()).name() << std::endl;
    //LOG(INFO) << key <<" "<< datum.id() <<" "<< datum.channels() <<" "<< datum.width() <<" "<< datum.height();
    datum.SerializeToString(&value);
    batch->Put(string(key), value);
	count ++;
    if (count % 1000 == 0) {
      db->Write(leveldb::WriteOptions(), batch);
      LOG(INFO) << "Processed " << count << " files.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    LOG(INFO) << "Processed " << count << " files.";
  }

  delete batch;
  delete db;
  return 0;
}
