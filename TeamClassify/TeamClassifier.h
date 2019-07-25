#pragma once
#include "Common.h"
#include "../ObjectDetectionBasketballGame/ITeamClassifier.h"

namespace TeamClassify
{
	/**
	 * Handles team classification.
	 */
	class TeamClassifier : public ITeamClassifier
	{
	public:
		enum
		{
			CODE_OK = 0, /**< Indicates success */
			CODE_ERROR, /**< Generic error */
			CODE_BAD_ARG, /**< One or more arguments are invalid */
		};

		/**
		 * Constructor
		 */
		TeamClassifier();

		/** Virtual destructor */
		virtual ~TeamClassifier();

		/**
		 * @brief Consumes a frame and processing to get team classification information.
		 * @return 0 on success, non-zero on error
		 */
		virtual int ProcessFrame(FrameProcParams* params);

		/**
		 * @brief Retrieves the result of the last processed frame.
		 * @param[out] results stores the result
		 */
		virtual void GetFrameProcResult(FrameData& result);

		/**
		 * @brief Gets the teams identified in the last processed frame.
		 * @param[out] teamA stores team A info
		 * @param[out] teamB stores team B info
		 */
		virtual void GetTeams(Team& teamA, Team& teamB);

	protected:		

		/**
		 * @brief Gets the general color of the player's jersey.
		 * The color obtained isn't necessarily the dominant color of the jersey but a color
		 * that can be associated with the player ROI. Under the hood k-means clustering
		 * is used to get the output color.
		 *
		 * @param[in] frame the current frame containing the player ROI
		 * @param[in] playerRect the player ROI
		 * @param[out] stores the ROI used to get the color
		 * @param[out] bgr stores the color associated to the player
		 * @return true if successful, false otherwise
		 */
		static bool GetPlayerBaseColor(const cv::Mat& frame, const cv::Rect playerRect, cv::Rect &roi, cv::Scalar& bgr);

		/**
		 * @brief Finds the index of the team the color is best associated with.
		 * @param[in] baseColor base color of the player(s)
		 * @retval 0 for team 0
		 * @retval 1 for team 1
		 * @retval -1 if there are no teams identified yet
		 * @retval -2 if cannot associate to with any known teams.
		 */
		int FindTeamIndex(cv::Scalar baseColor) const;

		/**
		 * @brief Adds a new team using the specified base color.
		 * @param[in] baseColor base color of the player(s) to add
		 * @return the index the new team profile was added to, or -1 on error
		 */
		int AddNewTeam(cv::Scalar baseColor);

		/**
		 * @brief Attempts to find a new team to associate a (player) base color and if no match is
		 * found will then automatically attempt to add a new team with the specified color.
		 * This function calls #FindTeamIndex and #AddNewTeam (if needed). After a matching team is
		 * found, the player's BoxProp::teamIdx is set.
		 * @param boxProp the player's box properties
		 * @param[in] newTeamBaseColor base color of the player(s) to add
		 * @return the index the new team profile was added to, or -1 on error
		 */
		int FindAddTeam(BoxProps &boxProps, cv::Scalar newTeamBaseColor);

		std::vector<FrameData> frameCtxts; /**< Stores relevant information for each frame */
		TeamClassify::Common::ProfStats profProcGetPlayerBaseColor; /**< Stores profiling stats for #GetPlayerBaseColor */
		TeamProps teamProps[2]; /**< Team properties */
		Team lastTeams[2]; /**< Stores last teams */
		const double DIST_THRESH = 75; /**< Threshold to associate a color to a team */
		long frameIdx; /**< Used to track frame, first is 1 */
	};
}

