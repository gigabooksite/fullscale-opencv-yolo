#ifndef COURTDETECT_H
#define COURTDETECT_H


#include <opencv2/opencv.hpp>

class CourtDetect
{
public:
	CourtDetect(const std::string& winName = "Court Detection");
	CourtDetect(const std::string& winName,	const std::string& settingsFile);
	virtual ~CourtDetect();

	bool setCourt(const std::string courtFile);
	cv::Mat getCourt();
	bool setSettingsFile(const std::string settingsFile);
	void projectPosition(cv::Mat& court, cv::Point2f position, cv::Scalar teamColor);
	void calibratePoints(const std::string& courtImage, const cv::Mat& frame);
	bool isCalibrated();
private:
	std::string winName;
	bool calibrated;
	std::string courtFile;
	cv::Mat intrinsics, distortion;
	cv::Mat court;

	std::vector<cv::Point2f> framePoints;
	std::vector<cv::Point2f> courtPoints;

	void markClickedPoint(cv::Mat& image, const size_t points, int x, int y);
	static void onMouseClickFrame(int event, int x, int y, int flags, void* param);
	static void onMouseClickCourt(int event, int x, int y, int flags, void* param);
};

#endif