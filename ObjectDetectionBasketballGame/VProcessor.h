#ifndef VPROCESSOR_H
#define VPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "MatQueue.h"
#include "CourtDetect.h"
#include "ITeamClassifier.h"

class VProcessor
{
public:
	VProcessor(MatQueue& in, MatQueue& out, TeamClassify::ITeamClassifier *tc);
	~VProcessor();
	void operator()();
private:
	MatQueue& _inFrames;
	MatQueue& _outFrames;
	TeamClassify::ITeamClassifier* teamClassifier;
	cv::dnn::Net _net;
	std::vector<std::string> _classes;
	std::vector<cv::Scalar> _colors;

	std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);
	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, const cv::Ptr<cv::dnn::Layer> lastLayer);
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, int teamIdx, int boxIdx);
	void assignTeams(TeamClassify::ITeamClassifier::FrameProcParams &fpp, std::vector<int>& teams);

	static const int m_inpWidth;
	static const int m_inpHeight;
	static const float m_confThreshold;
	static const float m_nmsThreshold;

	static const cv::String m_modelConfiguration;
	static const cv::String m_modelWeights;
	static const std::string m_classesFile;

	CourtDetect courtDetect;
};

#endif

