#pragma once

#include <opencv2/opencv.hpp>

namespace TeamClassify
{
/**
 * Provides interfaces for a team classifier
 */
class ITeamClassifier
{
public:
	/**
	 * Represents group of parameters needed by team classifier to process a frame
	 */
	struct FrameProcParams
	{
		cv::Mat frame; /**< frame to process */
		std::vector<cv::Rect> boxes; /**< a set of bounding boxes input to NMS */
		std::vector<int> indices; /**< the kept indices of bboxes after NMS */
		std::vector<std::string> classNames; /**< names of classes */
		std::vector<int> classIds; /**< the class IDs */
		std::vector<float> confidence; /**< confidence value */
	};

	/**
	 * Represents a detected bounding box output from the network.
	 */
	struct BoxProps
	{
		BoxProps()
		{
			classId = -1;
			confidence = 0.0;
			box = cv::Rect(0, 0, 0, 0);
			isPlayer = false;
			genBaseColor = cv::Scalar(0, 0, 0);
			teamIdx = -1;
		};
		int classId; /**< class ID */
		float confidence; /**< confidence value */
		cv::Rect box; /**< bounding box */
		bool isPlayer; /**< indicates of bbox is associated with a player */
		cv::Scalar genBaseColor; /**< General base color associated with this player */
		int teamIdx; /**< index used to associate the player with a team */
	};

	/**
	 * Groups information about the frame processed.
	 */
	struct FrameData
	{
		std::vector<BoxProps> boxes; /**< detected boxes in the the frame */
		std::vector<int> playerIndices; /**< indices of the player boxes */
	};

	/**
	 * @brief Consumes and processes a frame.
	 * @param params frame input parametesr
	 * @return 0 on success, or an error code specific to the classifier
	 */
	virtual int ProcessFrame(FrameProcParams* params) = 0;
};
}

