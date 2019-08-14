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

std::vector<cv::Point2f> framePoints;
std::vector<cv::Point2f> courtPoints;

#ifdef COURT_DETECT_ENABLED
const std::string courtWindow = "Court";
const std::string frameWindow = "Frame";

void onMouseClickFrame(int event, int x, int y, int flags, void* param)
{
	if (framePoints.size() < 4 && event == cv::EVENT_FLAG_LBUTTON)
	{
		framePoints.push_back(cv::Point2f(x, y));
		std::cout << "Frame corner " << x << "," << y << " captured\n";
	}
	else if (framePoints.size() == 4)
	{
		std::cout << "Finished capturing frame corners\n";
		cv::destroyWindow(frameWindow);
	}
}

void onMouseClickCourt(int event, int x, int y, int flags, void* param)
{
	if (courtPoints.size() < 4 && event == cv::EVENT_FLAG_LBUTTON)
	{
		courtPoints.push_back(cv::Point2f(x, y));
		std::cout << "Court corner " << x << "," << y << " captured\n";
	}
	else if (courtPoints.size() == 4)
	{
		std::cout << "Finished capturing court corners\n";
		cv::destroyWindow(courtWindow);
	}
}

void calibratePoints(const std::string& source)
{
	std::cout << "Click court and frame corners\n";

	// get court points
	cv::Mat court = cv::imread("courtdetect/court.png");

	cv::namedWindow(courtWindow);
	cv::setMouseCallback(courtWindow, onMouseClickCourt);
	cv::putText(court, "Click court corners", cv::Point(0, 25), 1, 2, cv::Scalar(0, 255, 0), 2);
	cv::imshow(courtWindow, court);

	// get frame points
	cv::VideoCapture cap;
	cap.open(source);
	cv::Mat frame;
	cap.read(frame);

	cv::namedWindow(frameWindow);
	cv::setMouseCallback(frameWindow, onMouseClickFrame);
	cv::putText(frame, "Click frame corners", cv::Point(0, 25), 1, 2, cv::Scalar(0, 255, 0), 2);
	cv::imshow(frameWindow, frame);

	cv::waitKey();
}
#endif

int main(int argc, char* argv[])
{
	MatQueue capture;
	MatQueue display;

#ifdef COURT_DETECT_ENABLED
	cv::String source = "courtdetect/video.mp4";
	calibratePoints(source);

	if (framePoints.size() < 4 || courtPoints.size() < 4)
	{
		std::cout << "ERROR failed to click all corners\n";
		return -1;
	}
#else
	cv::String source = "yolo/XWtjl9fI9pY_clip_11.mp4"; 
#endif
	
	ITeamClassifier* teamClassifer = TeamClassifierFactory::CreateTeamClassifier("dummy");
	VReader reader(capture, source);
	VProcessor processor(capture, display, teamClassifer, framePoints, courtPoints);
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

