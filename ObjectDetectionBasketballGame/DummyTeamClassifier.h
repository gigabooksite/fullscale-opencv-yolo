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
	};
}


