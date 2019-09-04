#ifndef COURT_STITCHER_H
#define COURT_STITCHER_H

#include <opencv2/opencv.hpp>

#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>

class CourtStitcher
{
public:
	CourtStitcher();
	virtual ~CourtStitcher();

	void calibrate(const std::vector<cv::Mat>& frames);
	cv::Mat stitch(const std::vector<cv::Mat>& frames);
	bool isCalibrated() const;
private:
	bool calibrated;
	float warped_image_scale;
	cv::Ptr<cv::WarperCreator> warper_creator;
	cv::Ptr<cv::detail::RotationWarper> warper;
	std::vector<cv::detail::CameraParams> cameras;
	cv::Ptr<cv::detail::ExposureCompensator> compensator;

	std::vector<cv::Point> corners;
	std::vector<cv::UMat> masks_warped;
	std::vector<cv::Size> full_img_sizes;
	std::vector<cv::Size> sizes;
};

#endif //COURT_STITCHER_H