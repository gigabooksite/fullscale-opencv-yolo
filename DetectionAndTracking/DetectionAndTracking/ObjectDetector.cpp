#include "ObjectDetector.h"

#include <opencv2/core/types_c.h>

const int ObjectDetector::m_inpWidth = 288;        // Width of network's input image
const int ObjectDetector::m_inpHeight = 288;       // Height of network's input image

#define YOLO
#ifdef YOLO
const cv::String ObjectDetector::m_modelConfiguration = "yolo/basketball-yolov3-tiny.cfg";
const cv::String ObjectDetector::m_modelWeights = "yolo/Weights/basketball-yolov3-tiny_7000.weights";
const std::string ObjectDetector::m_classesFile = "yolo/basketball.names";
#else //MobileNetSSD
const cv::String ObjectDetector::m_modelConfiguration = "mobilenetSSD/deploy.prototxt";
const cv::String ObjectDetector::m_modelWeights = "mobilenetSSD/deploy.caffemodel";
const std::string ObjectDetector::m_classesFile = "mobilenetSSD/classes.txt";
#endif

ObjectDetector::ObjectDetector()
{
}


ObjectDetector::~ObjectDetector()
{
}

void ObjectDetector::initialize()
{
	_net = cv::dnn::readNet(m_modelConfiguration, m_modelWeights);

	_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void ObjectDetector::detectObjects(cv::Mat& frame, std::vector<cv::Mat>& outs)
{
	cv::Mat blob;
	std::vector<cv::String> names = _net.getUnconnectedOutLayersNames();

	// Create a 4D blob from a frame.
	cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cvSize(m_inpWidth, m_inpHeight), cv::Scalar(0, 0, 0), true, false);

	//Sets the input to the network
	_net.setInput(blob);

	// Runs the forward pass to get output of the output layers
	_net.forward(outs, names);
}

cv::Ptr<cv::dnn::Layer> ObjectDetector::getOutputLayer()
{
	std::vector<cv::String> lnames = _net.getLayerNames();
	
	return _net.getLayer(static_cast<unsigned int>(lnames.size()));
}
