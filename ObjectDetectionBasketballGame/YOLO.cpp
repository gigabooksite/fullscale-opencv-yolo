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

std::atomic<bool> quit = false;

int main(int argc, char* argv[])
{
	MatQueue capture;
	MatQueue display;

	//VReader reader(capture, "yolo/XWtjl9fI9pY_clip_11.mp4");
	VReader reader(capture, "courtdetect/video.mp4");
	VProcessor processor(capture, display);
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

