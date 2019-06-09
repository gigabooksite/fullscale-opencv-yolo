#ifndef MATQUEUE_H
#define MATQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

#include <opencv2/opencv.hpp>

struct MyMat
{
public:
	cv::Mat mat;
	double fps;
	int width;
	int height;
};

class MatQueue
{
public:
	MatQueue();
	virtual ~MatQueue();

	MyMat pop();
	void push(MyMat frame);
private:
	std::queue<MyMat> _queue;
	std::mutex _mutex;
	std::condition_variable _cond;
};



#endif