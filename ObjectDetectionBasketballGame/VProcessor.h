#ifndef VPROCESSOR_H
#define VPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "MatQueue.h"
#include "CourtDetect.h"
#include "ITeamClassifier.h"
#include "ObjectDetector.h"

#ifdef TRACKING_ENABLED
#include "ObjectTracker.h"
#endif

class VProcessor
{
public:
	VProcessor(MatQueue& in, MatQueue& out, TeamClassify::ITeamClassifier *tc, std::string frameSource);
	~VProcessor();
	void operator()();
private:
	MatQueue& _inFrames;
	MatQueue& _outFrames;
	int trackCtr;
	TeamClassify::ITeamClassifier* teamClassifier;
	cv::dnn::Net _net;
	std::vector<std::string> _classes;
	std::vector<cv::Scalar> _colors;

	std::vector<int> _classIds;
	std::vector<float> _confidences;
	std::vector<cv::Rect> _boxes;
	std::vector<int> _indices;

	ObjectDetector _detector;
#ifdef TRACKING_ENABLED
	ObjectTracker _tracker;
#endif

	std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);
	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, const cv::Ptr<cv::dnn::Layer> lastLayer);
	void drawPred(cv::Mat& frame, std::vector<int>& teams);
	void classifyPlayer(cv::Mat& frame, std::vector<int>& teams);
	void assignTeams(TeamClassify::ITeamClassifier::FrameProcParams &fpp, std::vector<int>& teams);

	static const int m_inpWidth;
	static const int m_inpHeight;
	static const float m_confThreshold;
	static const float m_nmsThreshold;
	static const int MAX_TRACK_COUNT;

	static const cv::String m_modelConfiguration;
	static const cv::String m_modelWeights;
	static const std::string m_classesFile;
#ifdef COURT_DETECT_ENABLED
	CourtDetect courtDetect;
#endif
};

#endif

