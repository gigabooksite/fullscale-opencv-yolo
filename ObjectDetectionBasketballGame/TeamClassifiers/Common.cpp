#include "Common.h"
using namespace TeamClassify;

void Common::GetPerf(ProfStats& stats, int64 ref)
{
	double elapsedSec = (cv::getTickCount() - ref) / cv::getTickFrequency();
	stats.count++;
	stats.totalTime += elapsedSec;
	if (stats.maxTime < elapsedSec)
	{
		stats.maxTime = elapsedSec;
	}
	if (stats.minTime > elapsedSec || 0 == stats.minTime)
	{
		stats.minTime = elapsedSec;
	}
}

void Common::DrawKmeansImage(cv::Mat& rgbImageOut, cv::Mat& labels, const cv::Mat& centers, int height, int width)
{
	std::cout << "===\n";
	std::cout << "labels: " << labels.rows << " " << labels.cols << std::endl;
	std::cout << "centers: " << centers.rows << " " << centers.cols << std::endl;
	assert(labels.type() == CV_32SC1);
	assert(centers.type() == CV_32FC1);

	cv::Mat rgb_image(height, width, CV_8UC3);
	cv::MatIterator_<cv::Vec3b> rgb_first = rgb_image.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> rgb_last = rgb_image.end<cv::Vec3b>();
	cv::MatConstIterator_<int> label_first = labels.begin<int>();

	cv::Mat centers_u8;
	centers.convertTo(centers_u8, CV_8UC1, 255.0);
	cv::Mat centers_u8c3 = centers_u8.reshape(3);

	while (rgb_first != rgb_last) {
		const cv::Vec3b& rgb = centers_u8c3.ptr<cv::Vec3b>(*label_first)[0];
		*rgb_first = rgb;
		++rgb_first;
		++label_first;
	}
	rgbImageOut = rgb_image;
	//cv::imshow("tmp", rgb_image);
	//cv::imwrite("/Users/kumada/Data/graph-cut/output/result.jpg", rgb_image);
	//cv::waitKey();
}
