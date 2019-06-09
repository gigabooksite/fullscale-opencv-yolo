#include "VReader.h"

#include <atomic>

#include <opencv2/opencv.hpp>

extern std::atomic<bool> quit;

VReader::VReader(MatQueue& mat, const cv::String& inFile) : _frames(mat)
{
	_cap.open(inFile);
	//_cap.open(0); //camera
}


VReader::~VReader()
{
	_cap.release();
}

void VReader::operator()()
{
	if (_cap.isOpened() == false)
	{
		std::cout << "ERROR opening file" << std::endl;
	}
	
	MyMat myFrame;
	myFrame.width = int(_cap.get(cv::CAP_PROP_FRAME_WIDTH));
	myFrame.height = int(_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	myFrame.fps = _cap.get(cv::CAP_PROP_FPS);

	cv::Mat frame;
	do
	{
		if ((_cap.read(frame) == false) || (frame.empty()))
		{
			std::cout << "ERROR reading frame" << std::endl;
			break;
		}

		myFrame.mat = frame.clone();
		_frames.push(myFrame);
	} while (!quit);

	//Signal end of frame reading
	myFrame.width = 0;
	myFrame.mat = frame;
	_frames.push(myFrame);
}
