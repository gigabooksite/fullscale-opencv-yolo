// YOLO.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <opencv2/opencv.hpp>

#include <iostream>
#include <atomic>
#include <thread>

#include "MatQueue.h"
#include "VReader.h"
#include "VProcessor.h"
#include "VWriter.h"
#include "TeamClassifierFactory.h"

using namespace TeamClassify;
std::atomic<bool> quit = false;

int main(int argc, char* argv[])
{
	MatQueue capture;
	MatQueue display;

#ifdef COURT_DETECT_ENABLED
	cv::String source = "courtdetect/video.mp4";
#else
	cv::String source = "yolo/XWtjl9fI9pY_clip_11.mp4"; 
#endif
	
	ITeamClassifier* teamClassifer = TeamClassifierFactory::CreateTeamClassifier("dummy");
	VReader reader(capture, source);
	VProcessor processor(capture, display, teamClassifer);
	VWriter writer(display, "output.avi");

	std::thread t1(reader);
	std::thread t2(processor);
	std::thread t3(writer);

	while (!quit) { }

	t1.join();
	t2.join();
	t3.join();

	cv::waitKey(1);

	if (nullptr != teamClassifer) delete teamClassifer;

	return 0;
}

