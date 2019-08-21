#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

const int minImagePoints = 4;
const int maxImagePoints = 15;
const std::string courtWindow = "Court";
const std::string frameWindow = "Frame";

const cv::Scalar green(0, 255, 0);
const cv::Scalar black(0, 0, 0);

std::vector<cv::Point2f> framePoints, courtPoints;
cv::Point2f framePoint, courtPoint;

void markClickedPoint(cv::Mat& image, const size_t points, int x, int y)
{
	cv::circle(image, cv::Point(x, y), 1, black, 5);
	cv::circle(image, cv::Point(x, y), 1, green, 2);
	cv::putText(image, std::to_string(points), cv::Point(x - 5, y - 10), 1, 1, black, 5);
	cv::putText(image, std::to_string(points), cv::Point(x - 5, y - 10), 1, 1, green, 2);
}

void onMouseClickFrame(int event, int x, int y, int flags, void* param)
{
	if (framePoints.size() < maxImagePoints && event == cv::EVENT_FLAG_LBUTTON)
	{
		framePoints.push_back(cv::Point2f(x, y));
		cv::Mat& tempFrame = *((cv::Mat*)(param));
		markClickedPoint(tempFrame, framePoints.size(), x, y);

		std::cout << "Frame point " << x << "," << y << " captured\n";
		cv::imshow(frameWindow, tempFrame);
	}
	else if (framePoints.size() == maxImagePoints)
	{
		std::cout << "Finished capturing frame points\n";
		cv::destroyWindow(frameWindow);
	}
}

void onMouseClickCourt(int event, int x, int y, int flags, void* param)
{
	if (courtPoints.size() < maxImagePoints && event == cv::EVENT_FLAG_LBUTTON)
	{
		courtPoints.push_back(cv::Point2f(x, y));
		cv::Mat& tempCourt = *((cv::Mat*)(param));
		markClickedPoint(tempCourt, courtPoints.size(), x, y);

		std::cout << "Court point " << x << "," << y << " captured\n";
		cv::imshow(courtWindow, tempCourt);
	}
	else if (courtPoints.size() == maxImagePoints)
	{
		std::cout << "Finished capturing court points\n";
		cv::destroyWindow(courtWindow);
	}
}

void onMouseDrag(int event, int x, int y, int flags, void* param)
{
	if (event == cv::MouseEventTypes::EVENT_MOUSEMOVE)
	{
		cv::Point2f* sourcePoint = (cv::Point2f*)param;
		sourcePoint->x = (float)x;
		sourcePoint->y = (float)y;
	}
}

int main(int argc, char** argv)
{
	std::string	camSettingsFile = "cam1_data.xml",
				frameFile = "frame.jpg",
				courtFile = "court.png",
				videoFile = "video.png";

	cv::Mat frame = cv::imread(frameFile);
	cv::Mat court = cv::imread(courtFile);

	if (frame.empty() || court.empty())
	{
		std::cout << "Error reading image/s";
		return -1;
	}

	std::cout << "Click frame and court points\n";
	std::cout << "Up to 15 points can be captured. Press ESC to finish with less than 15 points.\n";

	// get points of court
	cv::namedWindow(courtWindow);
	cv::Mat tempCourt = court.clone();
	cv::putText(tempCourt, "Click court points", cv::Point(0, 25), 1, 2, cv::Scalar(0, 255, 0), 2);
	cv::imshow(courtWindow, tempCourt);
	cv::setMouseCallback(courtWindow, onMouseClickCourt, &tempCourt);

	// get points of frame
	cv::Mat intrinsics, distortion, undistortedFrame;
	cv::FileStorage fs(camSettingsFile, cv::FileStorage::READ);
	fs["camera_matrix"] >> intrinsics;
	fs["distortion_coefficients"] >> distortion;

	cv::undistort(frame, undistortedFrame, intrinsics, distortion);

	cv::resize(undistortedFrame, frame, cv::Size(800, 600));
	cv::namedWindow(frameWindow);
	cv::resize(undistortedFrame, frame, cv::Size(800, 600));

	cv::Mat tempFrame = frame.clone();
	cv::putText(tempFrame, "Click frame points", cv::Point(0, 25), 1, 2, cv::Scalar(0, 255, 0), 2);
	cv::imshow(frameWindow, tempFrame);
	cv::setMouseCallback(frameWindow, onMouseClickFrame, &tempFrame);

	cv::waitKey();
	cv::destroyWindow(courtWindow);
	cv::destroyWindow(frameWindow);

	if (framePoints.size() < minImagePoints || courtPoints.size() < minImagePoints)
	{
		std::cout << "ERROR must select at least 4 points for each.\n";
		return -1;
	}
	else if (framePoints.size() != courtPoints.size())
	{
		std::cout << "ERROR frame and court selected points must match.\n";
		return -1;
	}

	// match frame and court images
	std::cout << "Mapping frame and court images\n";
	cv::putText(frame, "Hover mouse on court.", cv::Point(0, 25), 1, 2, cv::Scalar(0, 255, 0), 2);
	cv::imshow(frameWindow, frame);

	cv::Mat homography = findHomography(framePoints, courtPoints, cv::RANSAC);
	cv::setMouseCallback(frameWindow, onMouseDrag, (void*)& framePoint);

	std::cout << "Finished mapping. Hover mouse over frame image to check.\n";
	while (1)
	{
		std::vector<cv::Point2f> srcVecP{ framePoint };
		std::vector<cv::Point2f> courtVecP{ courtPoint };
		perspectiveTransform(srcVecP, courtVecP, homography);

		courtPoint = courtVecP[0];

		cv::Mat courtClone = court.clone();
		cv::circle(courtClone, courtPoint, 3, cv::Scalar(0, 0, 255), 2, cv::LINE_8);

		cv::imshow(frameWindow, frame);
		cv::imshow(courtWindow, courtClone);

		char key = (char)cv::waitKey(30);
		if (key == 'q' || key == 27)
		{
			break;
		}
	}

	return 0;
}
