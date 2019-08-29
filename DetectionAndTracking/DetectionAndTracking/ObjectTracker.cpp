#include "ObjectTracker.h"



ObjectTracker::ObjectTracker()
{
}

ObjectTracker::~ObjectTracker()
{
}

void ObjectTracker::initialize(const std::vector<cv::Rect>& bboxes, cv::Mat& frame)
{
	for (auto bbox : bboxes)
	{
		// csrt - slow, good
		// kcf - average, average
		// mosse - fast, bad
		_multiTracker->add(cv::TrackerMOSSE::create(), frame, bbox);
	}
}

void ObjectTracker::reset()
{
	if (_multiTracker) _multiTracker->clear();
	_multiTracker = cv::MultiTracker::create();
}

void ObjectTracker::trackObjects(cv::Mat& frame, std::vector<cv::Rect>& bboxes)
{
	_multiTracker->update(frame);
	bboxes.clear();
	// get tracked objects
	for (unsigned i = 0; i < _multiTracker->getObjects().size(); i++)
	{
		bboxes.push_back(_multiTracker->getObjects()[i]);
	}
}