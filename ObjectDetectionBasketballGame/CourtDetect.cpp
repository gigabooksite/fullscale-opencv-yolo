#include "CourtDetect.h"

#include <opencv2/highgui.hpp>

CourtDetect::CourtDetect()
	: winName("Court Detection")
{
	cv::namedWindow(winName);
}

CourtDetect::CourtDetect(const std::string winName)
	: winName(winName)
{
}

CourtDetect::~CourtDetect()
{
}

CourtDetect::CourtDetect(const std::string winName, std::vector<cv::Point2f>& fPoints,
							std::vector<cv::Point2f>& cPoints, const std::string settingsFile,
							const std::string courtFile)
	: winName(winName), framePoints(fPoints), courtPoints(cPoints)
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

void CourtDetect::setFramePoints(const std::vector<cv::Point2f>& points)
{
	framePoints = points;
}

void CourtDetect::setCourtPoints(const std::vector<cv::Point2f>& points)
{
	courtPoints = points;
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

void CourtDetect::projectPosition(cv::Mat& court, cv::Point2f position)
{
	cv::Mat homography = findHomography(framePoints, courtPoints, cv::RANSAC);

	cv::Point2f courtPoint;
	std::vector<cv::Point2f> srcVecP{ position };
	std::vector<cv::Point2f> courtVecP{ courtPoint };

	cv::perspectiveTransform(srcVecP, courtVecP, homography);
	courtPoint = courtVecP[0];

	circle(court, courtPoint, 3, cv::Scalar(255), 1, cv::LINE_8);

	cv::imshow(winName, court);
	cv::waitKey(10);
}
