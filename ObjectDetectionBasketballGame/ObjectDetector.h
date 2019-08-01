#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

class ObjectDetector
{
public:
	ObjectDetector();
	virtual ~ObjectDetector();

	void initialize();
	void detectObjects(cv::Mat& frame, std::vector<cv::Mat>& outs);
	cv::Ptr<cv::dnn::Layer> getOutputLayer();

private:
	cv::dnn::Net _net;

	static const int m_inpWidth;
	static const int m_inpHeight;

	static const cv::String m_modelConfiguration;
	static const cv::String m_modelWeights;
	static const std::string m_classesFile;
};

#endif