# fullscale-opencv-yolo
Implementation of YOLOv3 using OpenCV - C++ library and Darknet - C Framework for YOLO Neural Network Training

# Solutions:
- [CalibrateCam](https://github.com/gigabooksite/fullscale-opencv-yolo/wiki/CalibrateCam-How-To) - utility function for getting camera intrinsics and distortion data
- CourtDetect2 - map a point in a basketball video to the equivalent point/position in a "flat/top-down-view court" image
- HSVRangeFinder - utility for finding the optimal HSV range for isolating a specific color
- ObjectDetectionBasketballGame - simple object detection program that detects player and ball in court
- TeamClassify - interfaces with ObjectDetectionBasketballGame to get each of the "player" blob and then associates it to one of the two teams

# Setup Guidelines
- OpenCV pre-compiled: https://sourceforge.net/projects/opencvlibrary/files/4.1.0/
            - opencv-4.1.0-vc14_vc15.exe
- OpenCV from Intel OpenVINO toolkit: https://software.intel.com/en-us/openvino-toolkit
            - w_openvino_toolkit_p_2019.1.148
- VS2019 Community: https://visualstudio.microsoft.com/
- https://www.opencv-srf.com/2017/11/install-opencv-with-visual-studio.html
- Manual OpenCV compilation needed for Object Tracking feature. Needs opencv_contrib
            - https://www.learnopencv.com/install-opencv3-on-windows/

# Dataset link:
- SPIRODOUME: https://sites.uclouvain.be/ispgroup/Softwares/SPIROUDOME
- APIDIS: https://sites.uclouvain.be/ispgroup/Softwares/APIDIS
- Videos converted to mp4: https://drive.google.com/drive/folders/1zPqdsXjPGO7MHSKoteczzE0SiBp6_Yzv?usp=sharing
