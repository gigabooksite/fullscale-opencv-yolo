#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

const std::string courtWindow = "Court";
const std::string frameWindow = "Frame";

std::vector<cv::Point2f> framePoints, courtPoints;
cv::Point2f framePoint, courtPoint;

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

	// get corner points of frame and court images
	std::cout << "Click frame and court corners\n";

	cv::namedWindow(courtWindow);
	cv::setMouseCallback(courtWindow, onMouseClickCourt);
	cv::imshow(courtWindow, court);

	cv::namedWindow(frameWindow);
	cv::setMouseCallback(frameWindow, onMouseClickFrame);
	cv::resize(frame, frame, cv::Size(800, 600));
	cv::imshow(frameWindow, frame);

	cv::waitKey();

	// match frame and court images
	std::cout << "Mapping frame and court images\n";
	cv::Mat intrinsics, distortion, undistortedFrame;
	cv::FileStorage fs(camSettingsFile, cv::FileStorage::READ);
	fs["camera_matrix"] >> intrinsics;
	fs["distortion_coefficients"] >> distortion;

	cv::undistort(frame, undistortedFrame, intrinsics, distortion);

	cv::resize(undistortedFrame, frame, cv::Size(800, 600));
	cv::imshow(frameWindow, frame);

	cv::Mat homography = findHomography(framePoints, courtPoints, cv::RANSAC);
	cv::setMouseCallback(frameWindow, onMouseDrag, (void*)& framePoint);

	std::cout << "Finished mapping\n";
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
