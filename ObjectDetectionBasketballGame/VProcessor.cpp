#include "VProcessor.h"

#include <atomic>
#include <fstream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>

extern std::atomic<bool> quit;
using namespace TeamClassify;

const int VProcessor::MAX_TRACK_COUNT = 5;
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
	  _inFrames(in)
	, _outFrames(out)
	, trackCtr(MAX_TRACK_COUNT)
	, teamClassifier(tc)
#ifdef COURT_DETECT_ENABLED
	, courtDetect("MyCourtDetection")
#endif
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

	_detector.initialize();

#ifdef COURT_DETECT_ENABLED
	courtDetect.setFramePoints(framePoints);
	courtDetect.setCourtPoints(courtPoints);
	courtDetect.setCourt("courtdetect/court.png");
#endif
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

		if (trackCtr == MAX_TRACK_COUNT)
		{
			cv::putText(frame, "DETECT", cv::Point(0, 20), 1, 1, cv::Scalar(0, 255, 0), 2);
			_detector.detectObjects(frame, outs);

			postprocess(frame, outs, _detector.getOutputLayer());
#ifdef TRACKING_ENABLED
			_tracker.reset();

			//initialize multitracker
			_tracker.initialize(_boxes, frame);

			trackCtr = 0;
#endif
		}
#ifdef TRACKING_ENABLED
		//update tracker
		cv::putText(frame, "TRACK", cv::Point(0, 20), 1, 1, cv::Scalar(0, 255, 0), 2);
		_tracker.trackObjects(frame, _boxes);
		++trackCtr;
#endif
		std::vector<int> teams;
		classifyPlayer(frame, teams);

		drawPred(frame, teams);

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
	_classIds.clear();
	_confidences.clear();
	_boxes.clear();
	_indices.clear();

	static int frameIdx = 1;

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

					_classIds.push_back(classIdPoint.x);
					_confidences.push_back((float)confidence);
					_boxes.push_back(cv::Rect(left, top, width, height));
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

				_classIds.push_back(idx);
				_confidences.push_back((float)confidence);
				_boxes.push_back(cv::Rect(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom));
			}

		}

	}
	
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	cv::dnn::NMSBoxes(_boxes, _confidences, m_confThreshold, m_nmsThreshold, _indices);

#if 0
	// For ground truth analysis.
	std::ostringstream pref;
	pref << "GT/GT_Fr" << std::setfill('0') << std::right << std::dec << std::setw(3) <<
		frameIdx << ".bmp";
	cv::imwrite(pref.str(), frame);
#endif

	frameIdx++;
}

//void VProcessor::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, int teamIdx, int boxIdx)
void VProcessor::drawPred(cv::Mat& frame, std::vector<int>& teams)
{
#ifdef COURT_DETECT_ENABLED
	cv::Mat courtCopy;
	courtCopy = courtDetect.getCourt().clone();
#endif
	for (size_t i = 0; i < _indices.size(); ++i)
	{
		int idx = _indices[i];
		cv::Rect box = _boxes[idx];

		int left = box.x;
		int top = box.y;
		int right = box.x + box.width;
		int bottom = box.y + box.height;

		// Draw a rectangle displaying the bounding box
		cv::Scalar teamColor;
		size_t teamIdx = teams[i];
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
			teamColor = cv::Scalar(0, 127, 255); // Unknown team / a ball
		}

		// Identify team by color
		if (teamIdx == 0 || teamIdx == 1)
		{
			cv::Mat overlay;
			frame.copyTo(overlay);

			ellipse(overlay, cv::Point(left + ((right - left) / 2), bottom - 10), cv::Size(60, 40), 0, 0, 360, teamColor, -1);

			cv::addWeighted(overlay, 0.3, frame, 0.7, 0, frame);
#if 0
			cv::RotatedRect rot = cv::RotatedRect(cv::Point((left + right) / 2, bottom), cv::Size(50, 16), 0);
			cv::ellipse(frame, rot, teamColor, 18);
#endif
		}
		else
		{
			rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), teamColor, 2);//cv::Scalar(0, 0, 255));
		}

		if (_classIds[idx] == 1)
		{
			cv::Point2f position;
			position.x = (float)box.x + (box.width / 2);
			position.y = (float)box.y + box.height;
#ifdef COURT_DETECT_ENABLED
			courtDetect.projectPosition(courtCopy, position, teamColor);
#endif
		}
	}
#if 0 //do not display label for now
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
#endif
}

void VProcessor::classifyPlayer(cv::Mat& frame, std::vector<int>& teams)
{
	// Setup and run team classifier.
	ITeamClassifier::FrameProcParams fpp;
	fpp.frame = frame;
	fpp.boxesNms = _boxes;
	fpp.indices = _indices;
	fpp.classNames = _classes;
	fpp.classIds = _classIds;
	fpp.confidence = _confidences;

	assignTeams(fpp, teams);
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