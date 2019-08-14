#ifndef COURTDETECT_H
#define COURTDETECT_H


#include <opencv2/opencv.hpp>

class CourtDetect
{
public:
	CourtDetect(const std::string& winName, const std::string& courtImage, const std::string& frameVideo);
	CourtDetect(const std::string& winName,
				std::vector<cv::Point2f>& fPoints,
				std::vector<cv::Point2f>& cPoints,
				const std::string settingsFile,
				const std::string courtFile);
	virtual ~CourtDetect();

	bool setCourt(const std::string courtFile);
	cv::Mat getCourt();
	bool setSettingsFile(const std::string settingsFile);
	void projectPosition(cv::Mat& court, cv::Point2f position, cv::Scalar teamColor);
private:
	std::string winName = "Court Detection";
	std::string courtFile;
	cv::Mat intrinsics, distortion;
	cv::Mat court;
};

#endif