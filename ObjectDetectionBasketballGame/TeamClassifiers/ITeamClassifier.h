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
		std::vector<cv::Rect> boxesNms; /**< a set of bounding boxes input to NMS */
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
			boxNms = cv::Rect(0, 0, 0, 0);
			isPlayer = false;
			genBaseColorRoi = cv::Rect(0, 0, 0, 0);
			genBaseColor = cv::Scalar(0, 0, 0);
			teamIdx = -1;
		};
		int classId; /**< class ID */
		float confidence; /**< confidence value */
		cv::Rect boxNms; /**< NMS bounding box */
		int boxNmsIdx; /**< NMS bounding box index */
		bool isPlayer; /**< indicates of bbox is associated with a player */
		cv::Rect genBaseColorRoi; /**< ROI processed to obtain #genBaseColor */
		cv::Scalar genBaseColor; /**< General base color associated with this player */
		int teamIdx; /**< index used to associate the player with a team (0 or 1) */
	};

	/**
	 * Groups information about the frame processed.
	 */
	struct FrameData
	{
		std::vector<BoxProps> boxProps; /**< detected boxes in the the frame */
		std::vector<int> playerBoxPropIndices; /**< indices of the player box prop */
	};

	/**
	 * Groups properties related to a team.
	 */
	struct TeamProps
	{
		TeamProps()
		{
			isValid = false;
			color = cv::Scalar(0, 0, 0);
			genBaseColor = cv::Scalar(0, 0, 0);
		}
		bool isValid; /**< Flag to indicate if a team is valid */
		cv::Scalar color; /**< Actual team color */
		cv::Scalar genBaseColor; /**< Base color used to associate players to team */
	};

	/**
	 * Represents a team
	 */
	struct Team
	{
		TeamProps props; /**< team properties */
		std::vector<int> playerBoxNmsIndices; /**< indices of box NMS (from the input FrameProcParams) of the players from this team */
	};

	virtual ~ITeamClassifier() {};

	/**
	 * @brief Consumes and processes a frame.
	 * @param params frame input parametesr
	 * @return 0 on success, or an error code specific to the classifier
	 */
	virtual int ProcessFrame(FrameProcParams* params) = 0;

	/**
	 * @brief Retrieves the result of the last processed frame.
	 * @param[out] results stores the result
	 */
	virtual void GetFrameProcResult(FrameData& result) = 0;

	/**
	 * @brief Gets the teams identified in the last processed frame.
	 * @param[out] teamA stores team A info
	 * @param[out] teamB stores team B info
	 */
	virtual void GetTeams(Team &teamA, Team &teamB) = 0;
};
}

