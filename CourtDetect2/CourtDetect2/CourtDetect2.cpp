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
						Point2f(454,388),
						Point2f(769,407),
						Point2f(1204,427),
						Point2f(1203,438),
						Point2f(1203,477),
						Point2f(1201,527),
						Point2f(1195,614),
						Point2f(1190,651),
						Point2f(430,541),
						Point2f(140,463),
						Point2f(170,455),
						Point2f(270,431),
						Point2f(344,413),
						Point2f(432,393),

						Point2f(388,417),
						Point2f(306,437),
						Point2f(497,409),
						Point2f(297,462),
						Point2f(638,419),
						Point2f(436,487),
						Point2f(720,425),
						Point2f(524,502),
						Point2f(812,431),
						Point2f(638,518),
						Point2f(1015,443),
						Point2f(913,552),
						Point2f(1120,473),
						Point2f(1103,521)
					},
					courtPoints{
						Point2f(29,	31),
						Point2f(320,31),
						Point2f(611,31),
						Point2f(611,47),
						Point2f(611,147),
						Point2f(611,221),
						Point2f(611,321),
						Point2f(611,337),
						Point2f(320,337),
						Point2f(29,	337),
						Point2f(29,	321),
						Point2f(29,	221),
						Point2f(29,	147),
						Point2f(29,	47),

						Point2f(73, 147),
						Point2f(73, 221),
						Point2f(137,91),
						Point2f(137,276),
						Point2f(258,91),
						Point2f(258,276),
						Point2f(320,91),
						Point2f(320,276),
						Point2f(382,91),
						Point2f(382,276),
						Point2f(503,91),
						Point2f(503,276),
						Point2f(568,147),
						Point2f(568,221)
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
	imshow("distorted", frame);
	waitKey();
	
	//Mat homography = findHomography(framePoints, courtPoints, RANSAC);
	//
	//while (1) {
	//	
	//		//cout << "Current mouse point = " << sourcePoint.x << "," << sourcePoint.y << endl;
	//		vector<Point2f> srcVecP{ sourcePoint };
	//		vector<Point2f> courtVecP{ courtPoint };
	//		perspectiveTransform(srcVecP, courtVecP, homography);

	//		courtPoint = courtVecP[0];

	//		Mat courtClone = court.clone();
	//		circle(courtClone, courtPoint, 3, Scalar(255), 1, LINE_8);

	//	imshow(kWinName, frame);
	//	imshow("Court", courtClone);

	//	char key = (char)waitKey(30);
	//	if (key == 'q' || key == 27)
	//	{
	//		break;
	//	}
	// }
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
