#include "VReader.h"

#include <atomic>

#include <opencv2/opencv.hpp>

extern std::atomic<bool> quit;

VReader::VReader(MatQueue& mat, const cv::String& inFile1, const cv::String& inFile2) : _frames(mat)
{
	_cap1.open(inFile1);
	_cap2.open(inFile2);
}


VReader::~VReader()
{
	_cap1.release();
	_cap2.release();
}

void VReader::operator()()
{
	if (_cap1.isOpened() == false)
	{
		std::cout << "ERROR opening file" << std::endl;
	}

	if (_cap2.isOpened() == false)
	{
		std::cout << "ERROR opening file" << std::endl;
	}
	
	MyMat myFrame;
	myFrame.width = int(_cap1.get(cv::CAP_PROP_FRAME_WIDTH));
	myFrame.height = int(_cap1.get(cv::CAP_PROP_FRAME_HEIGHT));
	myFrame.fps = _cap1.get(cv::CAP_PROP_FPS);

	cv::Mat frame, result;
	std::vector<cv::Mat> frames;
	do
	{
		frames.clear();
		if (_frames.size() == MAX_QUEUE_SIZE)
		{
			continue;
		}

		if (_cap1.isOpened())
		{
			if ((_cap1.read(frame) == false) || (frame.empty()))
			{
				std::cout << "ERROR reading frame1" << std::endl;
				break;
			}
			frames.push_back(frame.clone());
		}

		if (_cap2.isOpened())
		{
			if ((_cap2.read(frame) == false) || (frame.empty()))
			{
				std::cout << "ERROR reading frame2" << std::endl;
			}
			else
			{
				frames.push_back(frame.clone());

				if (!stitcher.isCalibrated())
				{
					stitcher.calibrate(frames);
				}

				result = stitcher.stitch(frames);
				result.convertTo(frame, CV_8UC3);
			}
		}
		myFrame.mat = frame.clone();
		_frames.push(myFrame);
	} while (!quit);

	//Signal end of frame reading
	myFrame.width = 0;
	myFrame.mat = frame.clone();
	_frames.push(myFrame);
}
