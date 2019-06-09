#ifndef VWRITER_H
#define VWRITER_H

#include "MatQueue.h"

#include <opencv2/opencv.hpp>

class VWriter
{
public:
	VWriter(MatQueue& mat, const cv::String& outFile);
	virtual ~VWriter();
	void operator()();
private:
	MatQueue& _frames;
	cv::VideoWriter _video;
};

#endif
