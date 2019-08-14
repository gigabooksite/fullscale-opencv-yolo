#include "CourtDetect.h"

#include <opencv2/highgui.hpp>

const int minImagePoints = 4;
const int maxImagePoints = 15;
const std::string courtWindow = "Court";
const std::string frameWindow = "Frame";

std::vector<cv::Point2f> framePoints;
std::vector<cv::Point2f> courtPoints;

void onMouseClickFrame(int event, int x, int y, int flags, void* param)
{
	if (framePoints.size() < maxImagePoints && event == cv::EVENT_FLAG_LBUTTON)
	{
		// @TODO mark clicked point in actual image?
		framePoints.push_back(cv::Point2f(x, y));
		std::cout << "Frame point " << x << "," << y << " captured\n";
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
		// @TODO mark clicked point in actual image?
		courtPoints.push_back(cv::Point2f(x, y));
		std::cout << "Court point " << x << "," << y << " captured\n";
	}
	else if (courtPoints.size() == maxImagePoints)
	{
		std::cout << "Finished capturing court points\n";
		cv::destroyWindow(courtWindow);
	}
}

void calibratePoints(const std::string& courtImage, const std::string& frameVideo)
{
	std::cout << "Click frame and court points\n";
	std::cout << "Up to 15 points can be captured. Press ESC to finish with less than 15 points.\n";

	// get court points
	cv::Mat court = cv::imread(courtImage);

	cv::Mat tempCourt = court.clone();
	cv::putText(tempCourt, "Click court points", cv::Point(0, 25), 1, 2, cv::Scalar(0, 255, 0), 2);
	cv::imshow(courtWindow, tempCourt);
	cv::namedWindow(courtWindow);
	cv::setMouseCallback(courtWindow, onMouseClickCourt);

	// get frame points
	cv::VideoCapture cap;
	cap.open(frameVideo);
	cv::Mat frame;
	cap.read(frame);

	cv::putText(frame, "Click frame points", cv::Point(0, 25), 1, 2, cv::Scalar(0, 255, 0), 2);
	cv::imshow(frameWindow, frame);
	cv::namedWindow(frameWindow);
	cv::setMouseCallback(frameWindow, onMouseClickFrame);

	cv::waitKey();
	cv::destroyWindow(courtWindow);
	cv::destroyWindow(frameWindow);

	if (framePoints.size() < minImagePoints || courtPoints.size() < minImagePoints)
	{
		std::cout << "ERROR must select at least 4 points for each.\n";
		std::exit(-1);
	}
	else if (framePoints.size() != courtPoints.size())
	{
		std::cout << "ERROR frame and court selected points must match.\n";
		std::exit(-1);
	}
}

CourtDetect::CourtDetect(const std::string& winName, const std::string& courtImage, const std::string& frameVideo)
	: winName(winName)
{
	calibratePoints(courtImage, frameVideo);
	setCourt(courtImage);
}

CourtDetect::~CourtDetect()
{
}

CourtDetect::CourtDetect(const std::string& winName, std::vector<cv::Point2f>& fPoints,
							std::vector<cv::Point2f>& cPoints, const std::string settingsFile,
							const std::string courtFile)
	: winName(winName)//, framePoints(fPoints), courtPoints(cPoints)
{
	cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
	if (fs.isOpened())
	{
		fs["camera_matrix"] >> intrinsics;
		fs["distortion_coefficients"] >> distortion;
	}
	cv::namedWindow(winName);

	court = cv::imread(courtFile);
}

bool CourtDetect::setCourt(const std::string courtFile)
{
	court = cv::imread(courtFile);
	if (court.empty())
	{
		return false;
	}
	return true;
}

cv::Mat CourtDetect::getCourt()
{
	return court;
}

bool CourtDetect::setSettingsFile(const std::string settingsFile)
{
	cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		return false;
	}
	fs["camera_matrix"] >> intrinsics;
	fs["distortion_coefficients"] >> distortion;
	return true;
}

void CourtDetect::projectPosition(cv::Mat& court, cv::Point2f position, cv::Scalar teamColor)
{
	cv::Mat homography = findHomography(framePoints, courtPoints, cv::RANSAC);

	cv::Point2f courtPoint;
	std::vector<cv::Point2f> srcVecP{ position };
	std::vector<cv::Point2f> courtVecP{ courtPoint };

	cv::perspectiveTransform(srcVecP, courtVecP, homography);
	courtPoint = courtVecP[0];

	circle(court, courtPoint, 3, teamColor, 2, cv::LINE_8);

	cv::imshow(winName, court);
	cv::waitKey(10);
}
