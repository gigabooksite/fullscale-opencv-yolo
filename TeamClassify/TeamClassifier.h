#pragma once
#include "Common.h"

namespace TeamClassify
{
	/**
	 * Handles team classification.
	 */
	class TeamClassifier
	{
	public:
		TeamClassifier() {};
		virtual ~TeamClassifier() {};

	protected:
		/**
		 * @brief Gets the general color of the player's jersey.
		 * The color obtained isn't necessarily the dominant color of the jersey but a color
		 * that can be associated with the player ROI. Under the hood k-means clustering
		 * is used to get the output color.
		 *
		 * @param[in] frame the current frame containing the player ROI
		 * @param[in] playerRect the player ROI
		 * @param[out] bgr stores the color associated to the player
		 * @return true if successful, false otherwise
		 */
		static bool GetPlayerBaseColor(const cv::Mat& frame, const cv::Rect playerRect, cv::Scalar& bgr);
	};
}

