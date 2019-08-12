#ifndef COURTDETECT_H
#define COURTDETECT_H


#include <opencv2/opencv.hpp>

class CourtDetect
{
public:
	CourtDetect();
	CourtDetect(const std::string winName = "Court Detection");
	CourtDetect(const std::string winName,
				std::vector<cv::Point2f>& fPoints,
				std::vector<cv::Point2f>& cPoints,
				const std::string settingsFile,
				const std::string courtFile);
	virtual ~CourtDetect();

	void setFramePoints(const std::vector<cv::Point2f>& points);
	void setCourtPoints(const std::vector<cv::Point2f>& points);
	bool setCourt(const std::string courtFile);
	cv::Mat getCourt();
	bool setSettingsFile(const std::string settingsFile);
	void projectPosition(cv::Mat& court, cv::Point2f position, cv::Scalar teamColor);
private:
	std::string winName;
	std::string courtFile;
	std::vector<cv::Point2f> framePoints;
	std::vector<cv::Point2f> courtPoints;
	cv::Mat intrinsics, distortion;
	cv::Mat court;
};

#endif