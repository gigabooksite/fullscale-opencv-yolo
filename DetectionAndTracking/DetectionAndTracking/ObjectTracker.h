#ifndef OBJECTTRACKER_H
#define OBJECTTRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

class ObjectTracker
{
public:
	ObjectTracker();
	virtual ~ObjectTracker();

	void initialize(const std::vector<cv::Rect>& bboxes, cv::Mat& frame);
	void reset();
	void trackObjects(cv::Mat& frame, std::vector<cv::Rect>& bboxes);
private:
	cv::Ptr<cv::MultiTracker> _multiTracker;
};

#endif