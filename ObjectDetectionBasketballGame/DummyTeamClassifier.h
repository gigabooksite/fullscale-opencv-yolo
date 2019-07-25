#pragma once
#include "ITeamClassifier.h"

namespace TeamClassify
{
	/**
	 * Implements a dummy team classifier
	 */
	class DummyTeamClassifier : public ITeamClassifier
	{
	public:		

		/**
		 * Constructor
		 */
		DummyTeamClassifier() {};

		/** Virtual destructor */
		virtual ~DummyTeamClassifier() {};

		/**
		 * @brief Consumes a frame and processing to get team classification information.
		 * @return 0 on success, non-zero on error
		 */
		virtual int ProcessFrame(FrameProcParams* params) { return 0; };

		/**
		 * @brief Retrieves the result of the last processed frame.
		 * @param[out] results stores the result
		 */
		virtual void GetFrameProcResult(FrameData& result) {};

		/**
		 * @brief Gets the teams identified in the last processed frame.
		 * @param[out] teamA stores team A info
		 * @param[out] teamB stores team B info
		 */
		virtual void GetTeams(Team& teamA, Team& teamB) {};
	};
}


