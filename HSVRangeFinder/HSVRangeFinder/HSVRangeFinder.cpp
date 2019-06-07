// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
//float confThreshold = 0.5; // Confidence threshold
//float nmsThreshold = 0.4;  // Non-maximum suppression threshold
//int inpWidth = 416;  // Width of network's input image
//int inpHeight = 416; // Height of network's input image
//vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
//void postprocess(Mat& frame, const vector<Mat>& out);
//
//// Draw the predicted bounding box
//void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

const string kWinName = "Deep learning object detection in OpenCV";
const int max_value_H = 360 / 2;
const int max_value = 255;
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;
static void on_low_H_thresh_trackbar(int, void*)
{
	low_H = min(high_H - 1, low_H);
	setTrackbarPos("Low H", kWinName, low_H);
}
static void on_high_H_thresh_trackbar(int, void*)
{
	high_H = max(high_H, low_H + 1);
	setTrackbarPos("High H", kWinName, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
	low_S = min(high_S - 1, low_S);
	setTrackbarPos("Low S", kWinName, low_S);
}
static void on_high_S_thresh_trackbar(int, void*)
{
	high_S = max(high_S, low_S + 1);
	setTrackbarPos("High S", kWinName, high_S);
}
static void on_low_V_thresh_trackbar(int, void*)
{
	low_V = min(high_V - 1, low_V);
	setTrackbarPos("Low V", kWinName, low_V);
}
static void on_high_V_thresh_trackbar(int, void*)
{
	high_V = max(high_V, low_V + 1);
	setTrackbarPos("High V", kWinName, high_V);
}

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);
	parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	// Load names of classes
	/*string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;*/
	// while (getline(ifs, line)) classes.push_back(line);

	// Give the configuration and weight files for the model
	//String modelConfiguration = "yolov3.cfg";
	//String modelWeights = "yolov3.weights";

	// Load the network
	//Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	//net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	//net.setPreferableTarget(DNN_TARGET_CPU);

	// Open a video file or an image file or a camera stream.
	string str, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, hsv, mask, res;

	try {

		outputFile = "yolo_out_cpp.avi";
		if (parser.has("image"))
		{
			cout << "Image Detected" << endl;
			// Open the image file
			str = parser.get<String>("image");
			frame = imread(str, IMREAD_COLOR);
			str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.jpg");
			outputFile = str;
		}
		else {
			cout << "Error" << endl;

		}

	}
	catch (...) {
		cout << "Could not open the input image/video stream" << endl;
		return 0;
	}

	namedWindow(kWinName, WINDOW_NORMAL);

	// Trackbars to set thresholds for HSV values
	createTrackbar("Low H", kWinName, &low_H, max_value_H, on_low_H_thresh_trackbar);
	createTrackbar("High H", kWinName, &high_H, max_value_H, on_high_H_thresh_trackbar);
	createTrackbar("Low S", kWinName, &low_S, max_value, on_low_S_thresh_trackbar);
	createTrackbar("High S", kWinName, &high_S, max_value, on_high_S_thresh_trackbar);
	createTrackbar("Low V", kWinName, &low_V, max_value, on_low_V_thresh_trackbar);
	createTrackbar("High V", kWinName, &high_V, max_value, on_high_V_thresh_trackbar);

	while (true) {
		// Stop the program if reached end of video
		if (frame.empty()) {
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
		}

		cout << "Frame not empty" << endl;
		// Declare the output variables
		Mat dst, cdst, cdstP;

		cvtColor(frame, hsv, COLOR_BGR2HSV);

		Scalar low_yellow = Scalar(44, 20, 100);
		Scalar high_yellow = Scalar(64, 20, 100);

		// define yellow HSV color range
		inRange(hsv, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), mask);

		// Edge detection
		Canny(frame, dst, 50, 200, 3);

		// Copy edges to the images that will display the results in BGR
		cvtColor(dst, cdst, COLOR_GRAY2BGR);
		cdstP = cdst.clone();

		// Standard Hough Line Transform
		vector<Vec2f> lines; // will hold the results of the detection
		HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
		// Draw the lines
		for (size_t i = 0; i < lines.size(); i++)
		{
			float rho = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			cv::line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
		}

		// Probabilistic Line Transform
		vector<Vec4i> linesP; // will hold the results of the detection
		HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
		// Draw the lines
		for (size_t i = 0; i < linesP.size(); i++)
		{
			Vec4i l = linesP[i];
			cv::line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
		}

		// Show results
		//imshow("Source", frame);
		//imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
		//imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
		//imshow("HSV", hsv);
		imshow(kWinName, mask);


		char key = (char)waitKey(30);
		if (key == 'q' || key == 27)
		{
			break;
		}
	}
	//// Create a 4D blob from a frame.
	//blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

	////Sets the input to the network
	//net.setInput(blob);

	//// Runs the forward pass to get output of the output layers
	//vector<Mat> outs;
	//net.forward(outs, net.getUnconnectedOutLayersNames());

	//// Remove the bounding boxes with low confidence
	//postprocess(frame, outs);

	//// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	//vector<double> layersTimes;
	//double freq = getTickFrequency() / 1000;
	//double t = net.getPerfProfile(layersTimes) / freq;
	//string label = format("Inference time for a frame : %.2f ms", t);
	//putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0));

	//// Write the frame with the detection boxes
	//Mat detectedFrame;
	//frame.convertTo(detectedFrame, CV_8U);
	//if (parser.has("image")) imwrite(outputFile, detectedFrame);
	//else video.write(detectedFrame);

	//imshow(kWinName, frame);
	return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
//void postprocess(Mat& frame, const vector<Mat>& outs)
//{
//	vector<int> classIds;
//	vector<float> confidences;
//	vector<Rect> boxes;
//
//	for (size_t i = 0; i < outs.size(); ++i)
//	{
//		// Scan through all the bounding boxes output from the network and keep only the
//		// ones with high confidence scores. Assign the box's class label as the class
//		// with the highest score for the box.
//		float* data = (float*)outs[i].data;
//		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
//		{
//			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
//			Point classIdPoint;
//			double confidence;
//			// Get the value and location of the maximum score
//			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
//			if (confidence > confThreshold)
//			{
//				int centerX = (int)(data[0] * frame.cols);
//				int centerY = (int)(data[1] * frame.rows);
//				int width = (int)(data[2] * frame.cols);
//				int height = (int)(data[3] * frame.rows);
//				int left = centerX - width / 2;
//				int top = centerY - height / 2;
//
//				classIds.push_back(classIdPoint.x);
//				confidences.push_back((float)confidence);
//				boxes.push_back(Rect(left, top, width, height));
//			}
//		}
//	}
//
//	// Perform non maximum suppression to eliminate redundant overlapping boxes with
//	// lower confidences
//	vector<int> indices;
//	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
//	for (size_t i = 0; i < indices.size(); ++i)
//	{
//		int idx = indices[i];
//		Rect box = boxes[idx];
//		drawPred(classIds[idx], confidences[idx], box.x, box.y,
//			box.x + box.width, box.y + box.height, frame);
//	}
//}
//
//// Draw the predicted bounding box
//void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
//{
//	//Draw a rectangle displaying the bounding box
//	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
//
//	//Get the label for the class name and its confidence
//	string label = format("%.2f", conf);
//	if (!classes.empty())
//	{
//		CV_Assert(classId < (int)classes.size());
//		label = classes[classId] + ":" + label;
//	}
//
//	//Display the label at the top of the bounding box
//	int baseLine;
//	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//	top = max(top, labelSize.height);
//	rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
//	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
//}