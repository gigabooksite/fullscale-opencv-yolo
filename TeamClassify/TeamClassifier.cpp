#include "TeamClassifier.h"

using namespace TeamClassify;

bool TeamClassifier::GetPlayerBaseColor(const cv::Mat& frame, const cv::Rect playerRect, cv::Scalar& bgr)
{
	bool success = false;

	if (!frame.empty())
	{
		// Focus on smaller portion of the blob - the jersey ROI.
		int boxRectNewWidth = playerRect.width / 4;
		int boxRectNewHeight = playerRect.height - (playerRect.height / 4);
		cv::Rect boxRectNew = cv::Rect(
			(playerRect.x + playerRect.width / 2) - boxRectNewWidth / 2,
			(playerRect.y + playerRect.height / 2) - boxRectNewHeight / 2,
			boxRectNewWidth,
			boxRectNewHeight);
		cv::Mat jerseyRoi = frame(boxRectNew);

		// Get HSV.
		cv::Mat roiHsv;
		cv::cvtColor(jerseyRoi, roiHsv, cv::COLOR_BGR2HSV);
		cv::Mat h, s, v;
		std::vector<cv::Mat> ch;
		cv::split(roiHsv, ch);
		h = ch[0];
		s = ch[1];
		v = ch[2];

		// Create and use mask to isolate player from background.
		cv::Mat h_filtered = (h < 181) & (h > 105);
		cv::Mat h_filtered_img; cv::copyTo(jerseyRoi, h_filtered_img, h_filtered);
#if 0
		// Test intermediate images.
		cv::Mat s_filtered = (s > 20);
		cv::Mat v_filtered = (v < 200);
		cv::Mat hs_filtered = h_filtered & s_filtered;
		cv::Mat hv_filtered = h_filtered & v_filtered;
		cv::Mat sv_filtered = s_filtered & v_filtered;
#endif

		// Reshape and reformat input image as required by cv::kmeans.
		cv::Mat kinImg = h_filtered_img;
		cv::Mat reshaped_image = kinImg.reshape(1, kinImg.cols * kinImg.rows);
		cv::Mat kin; reshaped_image.convertTo(kin, CV_32FC1, 1.0 / 255.0);

		// Setup parameters and invoke cv::kmeans.
		cv::Mat labels;
		cv::Mat centers;
		const int clusterNum = 1;
		cv::TermCriteria criteria{ cv::TermCriteria::COUNT, 50, 1 };
		cv::kmeans(kin, clusterNum, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);

		// Convert to 8-bit.
		cv::Mat centers_u8;
		centers.convertTo(centers_u8, CV_8UC1, 255.0);

		// Identify R, G and B values.
		uchar jcB = centers_u8.at<uchar>(0);
		uchar jcG = centers_u8.at<uchar>(1);
		uchar jcR = centers_u8.at<uchar>(2);

		if (1 < clusterNum)
		{
			// Alternative, crude but fast fast way to get jersey color.
			uchar* px = centers_u8.data;
			int first = 0, second = 0;
			int r = 0;
			for (; r < centers_u8.cols; ++r)
			{
				first += px[r];
			}
			for (; r < centers_u8.cols * 2; ++r)
			{
				second += px[r];
			}
			if (first < second)
			{
				jcB = centers_u8.at<uchar>(centers_u8.cols + 0);
				jcG = centers_u8.at<uchar>(centers_u8.cols + 1);
				jcR = centers_u8.at<uchar>(centers_u8.cols + 2);
			}
		}

		bgr = cv::Scalar(jcB, jcG, jcR);
		success = true;
	}

	return success;
}
