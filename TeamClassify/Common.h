#pragma once

#include <opencv2/opencv.hpp>

/**
 * Writes a cv::Mat into a BMP file using the variable name as filename.
 */
#define WRITE_BMP(pref, x) cv::imwrite(pref + std::string(#x ## ".bmp"), x)

namespace TeamClassify
{
	/**
	 * @brief Groups common functions and properties.
	 */
	class Common
	{
	public:
		/**
		 * Profiling stats
		 */
		struct ProfStats
		{
			ProfStats()
			{
				count = 0;
				totalTime = 0.0;
				averageTime = 0.0;
				maxTime = 0.0;
				minTime = 0.0;
			}
			unsigned long count; /**< Stores count */
			double totalTime; /**< Stores total time */
			double averageTime; /**< Stores avegage time */
			double maxTime; /**< Stores max time */
			double minTime; /**< Stores min time */
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
			};
			int classId; /**< class ID */
			float confidence; /**< confidence value */
			cv::Rect box; /**< bounding box */
		};

		/**
		 * Represents a frame and its properties.
		 */
		struct FrameContext
		{
			std::vector<BoxProps> boxes; /**< detected boxes in the the frame */
		};

		/**
		 * @brief Gets performance stats.
		 * Updates count, min, max and total values.
		 * @param stats profiling stats
		 * @param ref reference tick count obtained by cv::getTickCount(). 
		 */
		static void GetPerf(ProfStats& stats, int64 ref);

		/**
		 * @brief Draw image using new pixels values from k-means result.
		 * @param[out] rgbImageOut stores the redrawn image
		 * @param[in] labels labels output of k-means
		 * @param[in] centers centers output of k-means
		 * @param[in] height source image height
		 * @param[in] width source image width
		 */
		static void DrawKmeansImage(cv::Mat& rgbImageOut, cv::Mat& labels, const cv::Mat& centers, int height, int width);

	private:
		/**
		 * Default constructor.
		 */
		Common() {};

		/**
		 * Virtual destructor.
		 */
		virtual ~Common() {};

		/**
		 * Private copy constructor.
		 */
		Common(const Common&) = delete;

		/**
		 * Private assignment operator.
		 */
		Common& operator=(const Common&) = delete;
	};
}

