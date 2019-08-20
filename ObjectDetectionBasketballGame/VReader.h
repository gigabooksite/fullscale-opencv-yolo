#ifndef VREADER_H
#define VREADER_H

#include <opencv2/opencv.hpp>

#include "MatQueue.h"
#include "CourtStitcher.h"

class VReader
{
public:
	VReader(MatQueue& mat, const cv::String& inFile1, const cv::String& inFile2);
	virtual ~VReader();
	void operator()();
private:
	MatQueue& _frames;
	cv::VideoCapture _cap1;
	cv::VideoCapture _cap2;
	CourtStitcher stitcher;

	const size_t MAX_QUEUE_SIZE = 1000;
};

#endif
