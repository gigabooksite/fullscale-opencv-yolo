#include "VProcessor.h"

#include <atomic>
#include <fstream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>

extern std::atomic<bool> quit;
using namespace TeamClassify;

const int VProcessor::m_inpWidth = 288;        // Width of network's input image
const int VProcessor::m_inpHeight = 288;       // Height of network's input image
const float VProcessor::m_nmsThreshold = 0.4f;  // Non-maximum suppression threshold
const float VProcessor::m_confThreshold = 0.5f; // Confidence threshold

#define YOLO

#ifdef YOLO
const cv::String VProcessor::m_modelConfiguration = "yolo/basketball-yolov3-tiny.cfg";
const cv::String VProcessor::m_modelWeights = "yolo/Weights/basketball-yolov3-tiny_7000.weights";
const std::string VProcessor::m_classesFile = "yolo/basketball.names";
#else //MobileNetSSD
const cv::String VProcessor::m_modelConfiguration = "mobilenetSSD/deploy.prototxt";
const cv::String VProcessor::m_modelWeights = "mobilenetSSD/deploy.caffemodel";
const std::string VProcessor::m_classesFile = "mobilenetSSD/classes.txt";
#endif

std::vector<cv::Point2f>
framePoints{ cv::Point2f(56,782),
			cv::Point2f(615,442),
			cv::Point2f(1455,491),
			cv::Point2f(1409,1104),
};

std::vector<cv::Point2f>
courtPoints{cv::Point2f(28,	332),
			cv::Point2f(28,30),
			cv::Point2f(319,30),
			cv::Point2f(319,336),

};

VProcessor::VProcessor(MatQueue& in, MatQueue& out, ITeamClassifier* tc) :
	_inFrames(in),
	_outFrames(out),
	teamClassifier(tc),
	courtDetect("MyCourtDetection")
{
	cv::RNG rng(12345);

	// Load names of classes
	std::ifstream ifs(m_classesFile.c_str());
	std::string line;
	while (getline(ifs, line))
	{
		_classes.push_back(line);
		_colors.push_back(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
	}

	_net = cv::dnn::readNet(m_modelConfiguration, m_modelWeights);

	_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	courtDetect.setFramePoints(framePoints);
	courtDetect.setCourtPoints(courtPoints);
	courtDetect.setCourt("courtdetect/court.png");
}


VProcessor::~VProcessor()
{
}

void VProcessor::operator()()
{
	cv::Mat frame, blob;
	std::vector<cv::Mat> outs;

	static std::vector<cv::String> names = _net.getUnconnectedOutLayersNames();
	MyMat myFrame;
	do
	{
		myFrame = _inFrames.pop();
		frame = myFrame.mat;
		if (myFrame.width == 0)
		{
			myFrame.mat = frame.clone();
			_outFrames.push(myFrame);
			break;
		}

		// Create a 4D blob from a frame.
		cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cvSize(m_inpWidth, m_inpHeight), cv::Scalar(0, 0, 0), true, false);

		//Sets the input to the network
		_net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		_net.forward(outs, names);

		std::vector<cv::String> lnames = _net.getLayerNames();
		cv::Ptr<cv::dnn::Layer> outputLayer = _net.getLayer(static_cast<unsigned int>(lnames.size()));

		// Remove the bounding boxes with low confidence
		postprocess(frame, outs, outputLayer);

		std::vector<double> layersTimes;
		double freq = cv::getTickFrequency() / 1000;
		double t = _net.getPerfProfile(layersTimes) / freq;
		std::string label = cv::format("Inference time for a frame : %.2f ms", t);

		// Write the frame with the detection boxes
		frame.convertTo(frame, CV_8U);
		
		myFrame.mat = frame.clone();
		_outFrames.push(myFrame);

	} while (!quit);
}

std::vector<cv::String> VProcessor::getOutputsNames(const cv::dnn::Net& net)
{
	return net.getUnconnectedOutLayersNames();
}

void VProcessor::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, const cv::Ptr<cv::dnn::Layer> lastLayer)
{
	static int frameIdx = 1;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> teams;

	if (lastLayer->type.compare("Region") == 0)
	{
		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Scan through all the bounding boxes output from the network and keep only the
			// ones with high confidence scores. Assign the box's class label as the class
			// with the highest score for the box.
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIdPoint;
				double confidence;
				// Get the value and location of the maximum score
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > m_confThreshold)
				{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(cv::Rect(left, top, width, height));
				}
			}
		}
	}
	else if (lastLayer->type.compare("DetectionOutput") == 0)
	{
		std::ostringstream ss;
		//cv::Mat detectionMat(outs[i].size[2], outs[i].size[3], CV_32F, outs[i].ptr<float>());
		cv::Mat detection = outs.front();
		cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);

			if (confidence > m_confThreshold)
			{
				int idx = static_cast<int>(detectionMat.at<float>(i, 1));
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				classIds.push_back(idx);
				confidences.push_back((float)confidence);
				boxes.push_back(cv::Rect(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom));
			}

		}

	}
	
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);

	// Setup and run team classifier.
	ITeamClassifier::FrameProcParams fpp;
	fpp.frame = frame;
	fpp.boxesNms = boxes;
	fpp.indices = indices;
	fpp.classNames = _classes;
	fpp.classIds = classIds;
	fpp.confidence = confidences;
	assignTeams(fpp, teams);

	cv::Mat courtCopy;
	courtCopy = courtDetect.getCourt().clone();
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, teams[i], idx);

		if (classIds[idx] == 1)
		{
			cv::Point2f position;
			position.x = (float)box.x + (box.width / 2);
			position.y = (float)box.y + box.height;
			courtDetect.projectPosition(courtCopy, position);
		}
	}

#if 0
	// For ground truth analysis.
	std::ostringstream pref;
	pref << "GT/GT_Fr" << std::setfill('0') << std::right << std::dec << std::setw(3) <<
		frameIdx << ".bmp";
	cv::imwrite(pref.str(), frame);
#endif

	frameIdx++;
}

void VProcessor::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, int teamIdx, int boxIdx)
{
	// Draw a rectangle displaying the bounding box
	cv::Scalar teamColor;
	if (teamIdx == 0)
	{
		teamColor = cv::Scalar(255, 0, 0); // Blue team
	}
	else if (teamIdx == 1)
	{
		teamColor = cv::Scalar(0, 0, 255); // Red team
	}
	else
	{
		rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), _colors[classId], 2);//cv::Scalar(0, 0, 255));
	}

	// Identify team by color
	if (teamIdx == 0 || teamIdx == 1)
	{
		rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), teamColor, 2);
#if 0
		cv::RotatedRect rot = cv::RotatedRect(cv::Point((left + right) / 2, bottom), cv::Size(50, 16), 0);
		cv::ellipse(frame, rot, teamColor, 18);
#endif
	}

	// Get the box index and label for the class name and its confidence
	std::string label = cv::format("[%d]%.2f", boxIdx, conf);
	if (!_classes.empty())
	{
		CV_Assert(classId < (int)_classes.size());
		label = _classes[classId] + ":" + label;
	}

	// Display the label at the top of the bounding box
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = cv::max(top, labelSize.height);
	putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, _colors[classId], 2); //cv::Scalar(255, 255, 255));
}

void VProcessor::assignTeams(ITeamClassifier::FrameProcParams& fpp, std::vector<int>& teams)
{	
	if (!fpp.frame.empty() && 0 == teamClassifier->ProcessFrame(&fpp))
	{		
		// Assign players to teams.
		ITeamClassifier::Team teamA;
		ITeamClassifier::Team teamB;
		teams.clear();
		teamClassifier->GetTeams(teamA, teamB);
		for (size_t i = 0; i < fpp.indices.size(); ++i)
		{
			int idx = fpp.indices[i];
			int teamIdx = -1;
			if (fpp.classIds[idx] == 1)
			{
				if (teamA.props.isValid)
				{
					auto it = std::find(teamA.playerBoxNmsIndices.begin(), teamA.playerBoxNmsIndices.end(), idx);
					if (it != teamA.playerBoxNmsIndices.end())
					{
						teamIdx = 0;
					}
				}
				if (teamIdx == -1 && teamB.props.isValid)
				{
					auto it = std::find(teamB.playerBoxNmsIndices.begin(), teamB.playerBoxNmsIndices.end(), idx);
					if (it != teamB.playerBoxNmsIndices.end())
					{
						teamIdx = 1;
					}
				}
			}
			teams.push_back(teamIdx);
		}
	}
}