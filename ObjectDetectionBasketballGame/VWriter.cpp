#include "VWriter.h"

#include <atomic>
#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>

extern std::atomic<bool> quit;

#ifdef COURT_DETECT_ENABLED.
const cv::Size outputSize(1600, 1200);
#else
const cv::Size outputSize(1280, 720);
#endif

VWriter::VWriter(MatQueue& mat, const cv::String& outFile) : _frames(mat),
	_video(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 29, outputSize)
{
	
}


VWriter::~VWriter()
{
	cv::destroyAllWindows();
}

void VWriter::operator()()
{
	cv::Mat frame;
	MyMat myFrame;
	bool setprop = false;
	do
	{
		myFrame = _frames.pop();
		if (myFrame.width == 0)
		{
			break;
		}
		frame = myFrame.mat;
		if (!setprop)
		{
			_video.set(cv::CAP_PROP_FRAME_WIDTH, myFrame.width);
			_video.set(cv::CAP_PROP_FRAME_HEIGHT, myFrame.height);
			_video.set(cv::CAP_PROP_FPS, myFrame.fps);
			setprop = true;
		}

		cv::imshow("My Video", frame);
		_video.write(frame);
		if (cv::waitKey(1) == 27) //esc key
		{
			quit = true;
			break;
		}
	} while (!quit);

	quit = true;
}
