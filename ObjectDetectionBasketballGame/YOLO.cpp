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

#ifdef _WITH_TEAMCLASSIFIER
#include "..\TeamClassify\TeamClassifier.h"
#else
#include "DummyTeamClassifier.h"
#endif

using namespace TeamClassify;
std::atomic<bool> quit = false;

int main(int argc, char* argv[])
{
	MatQueue capture;
	MatQueue display;


#ifdef _WITH_TEAMCLASSIFIER
	VReader reader(capture, "yolo/XWtjl9fI9pY_clip_11.mp4");	
	TeamClassifier teamClassifier;
#else
	VReader reader(capture, "courtdetect/video.mp4");
	DummyTeamClassifier teamClassifier;
#endif

	VProcessor processor(capture, display, &teamClassifier);
	VWriter writer(display, "output.avi");

	std::thread t1(reader);
	std::thread t2(processor);
	std::thread t3(writer);

	while (!quit) { }

	t1.join();
	t2.join();
	t3.join();

	cv::waitKey(1);

	return 0;
}

