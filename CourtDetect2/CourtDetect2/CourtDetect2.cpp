#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

const string kWinName = "Court Detection";

void onMouseClickFrame(int event, int x, int y, int flags, void* param);

int main(int argc, char** argv)
{
	string	camSettingsFile = "cam1_data.xml",
			frameFile = "frame.jpg",
			courtFile = "court.png",
			videoFile = "video.png";
	Mat intrinsics, distortion, undistortedFrame;

	// These points were manually found by impementing a click handler on the window and outputing
	// the x,y position on the console using cout
	vector<Point2f>	framePoints{ 
						Point2f(1,399),
						Point2f(309,222),
						Point2f(745,245),
						Point2f(729,571),
					},
					courtPoints{
						Point2f(30,	320),
						Point2f(30,31),
						Point2f(320,31),
						Point2f(320,338),
					};
	Point2f sourcePoint, courtPoint;

	Mat frame = imread(frameFile);
	Mat court = imread(courtFile);
	FileStorage fs(camSettingsFile, FileStorage::READ);
	fs["camera_matrix"] >> intrinsics;
	fs["distortion_coefficients"] >> distortion;
	
	namedWindow(kWinName);
	setMouseCallback(kWinName, onMouseClickFrame, (void*)& sourcePoint);

	if (frame.empty() || court.empty()) {
		cout << "Error reading image";
		return 0;
	}

	undistort(frame, undistortedFrame, intrinsics, distortion);

	Mat resized;
	cv::resize(undistortedFrame, resized, Size(800, 600));
	imshow(kWinName, resized);

	
	Mat homography = findHomography(framePoints, courtPoints, RANSAC);
	
	while (1) {
		
			//cout << "Current mouse point = " << sourcePoint.x << "," << sourcePoint.y << endl;
			vector<Point2f> srcVecP{ sourcePoint };
			vector<Point2f> courtVecP{ courtPoint };
			perspectiveTransform(srcVecP, courtVecP, homography);

			courtPoint = courtVecP[0];

			Mat courtClone = court.clone();
			circle(courtClone, courtPoint, 3, Scalar(255), 1, LINE_8);

		imshow(kWinName, resized);
		imshow("Court", courtClone);

		char key = (char)waitKey(30);
		if (key == 'q' || key == 27)
		{
			break;
		}
	 }
	return 0;
}

void onMouseClickFrame(int event, int x, int y, int flags, void* param)
{
	// Capture the point coordinates, for now we will do it manually instead of feature-detection
	if (event == EVENT_FLAG_LBUTTON) {
		cout << "Point " << x << "," << y << " captured" << endl;
	}

	// Map where the mouse is hovering on the video frame to a separate "flat court" picture
	if (event == MouseEventTypes::EVENT_MOUSEMOVE)
	{
		Point2f* sourcePoint = (Point2f*) param;
		sourcePoint->x = (float)x;
		sourcePoint->y = (float)y;
	}
}
