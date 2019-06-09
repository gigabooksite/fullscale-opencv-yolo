#ifndef VREADER_H
#define VREADER_H

#include <opencv2/opencv.hpp>

#include "MatQueue.h"

class VReader
{
public:
	VReader(MatQueue& mat, const cv::String& inFile);
	virtual ~VReader();
	void operator()();
private:
	MatQueue& _frames;
	cv::VideoCapture _cap;
};

#endif
