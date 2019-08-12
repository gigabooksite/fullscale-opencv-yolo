#include "MatQueue.h"

MatQueue::MatQueue()
{
}


MatQueue::~MatQueue()
{
}

MyMat MatQueue::pop()
{
	std::unique_lock<std::mutex> mlock(_mutex);
	while (_queue.empty())
	{
		_cond.wait(mlock);
	}
	auto item = _queue.front();
	_queue.pop();
	return item;
}

void MatQueue::push(MyMat frame)
{
	std::unique_lock<std::mutex> mlock(_mutex);
	_queue.push(frame);
	mlock.unlock();
	_cond.notify_one();
}

size_t MatQueue::size() const
{
	return _queue.size();
}