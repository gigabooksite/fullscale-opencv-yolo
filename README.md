# fullscale-opencv-yolo
Implementation of YOLOv3 using OpenCV - C++ library and Darknet - C Framework for YOLO Neural Network Training

# Solutions:
- [CalibrateCam](https://github.com/gigabooksite/fullscale-opencv-yolo/wiki/CalibrateCam-How-To) - utility function for getting camera intrinsics and distortion data
- CourtDetect2 - manually map a set of points in a basketball game image to the equivalent point/position in a "flat/top-down-view court" image, and detect mouse location accordingly
- HSVRangeFinder - utility for finding the optimal HSV range for isolating a specific color
- ObjectDetectionBasketballGame - simple object detection program that detects player and ball in court
- ObjectDetectionBasketballGame\TeamClassifiers - contains implementations for associating each player "blob" to one of the two teams

# Setup Guidelines
- OpenCV
    - Pre-compiled:
        - https://sourceforge.net/projects/opencvlibrary/files/4.1.0/
        - opencv-4.1.0-vc14_vc15.exe
    - From Intel OpenVINO toolkit: 
        - https://software.intel.com/en-us/openvino-toolkit
        - w_openvino_toolkit_p_2019.1.148
    - Manual compilation
        - Needed for ObjectTracking feature using opencv_contrib
        - https://www.learnopencv.com/install-opencv3-on-windows/
		- Note: Check `BUILD_opencv_world` to generate necessary `.dll` file
- VS2019 Community: https://visualstudio.microsoft.com/
- OpenCV setup for VS2019: https://www.opencv-srf.com/2017/11/install-opencv-with-visual-studio.html

# Dataset link:
- SPIRODOUME: https://sites.uclouvain.be/ispgroup/Softwares/SPIROUDOME
- APIDIS: https://sites.uclouvain.be/ispgroup/Softwares/APIDIS
- Videos converted to mp4: https://drive.google.com/drive/folders/1zPqdsXjPGO7MHSKoteczzE0SiBp6_Yzv?usp=sharing

# Darknet Tips

## Build Darknet with VS2019 and OpenCV from OpenVINO toolkit

**Prerequisites:**
- Visual Studio 2019
- Intel OpenVINO Toolkit (w_openvino_toolkit_p_2019.1.148)

**Procedure:**
- Clone https://github.com/AlexeyAB/darknet.git.
- Open `darknet_no_gpu.sln`, from `darknet\build\darknet`, in VS2019.
- Set build config to Release x64.
- In project properties, change OpenCV include path to point to the one from the OpenVINO install directory.
	> **Example:** `C:\Program Files (x86)\IntelSWTools\openvino_2019.1.148\opencv\include`
- In project properties, change OpenCV library path to the one from the OpenVINO install directory.
	> **Example:** `C:\Program Files (x86)\IntelSWTools\openvino_2019.1.148\opencv\lib`
- Build `darknet_no_gpu` project. On successful build, the output files are placed in `darknet\build\darknet\x64`.

**(Optional) Set up environment setup script:**
- Download `setupvars.bat` from `fullscale-opencv-yolo\Darknet` and place it in `darknet\scripts`.
- Edit `setupvars.bat` and set `INTEL_OPENVINO_DIR` to the OpenVINO install directory.
	> **Example:** `set  "INTEL_OPENVINO_DIR=C:\Program Files (x86)\IntelSWTools\openvino_2019.1.148"`
- Open a terminal and run the script.
	```
	C:\>call darknet\scripts\setupvars.bat
	ECHO is off.
	----------------------------------------------------------------------------- 
	SUCCESS. Darknet environment initialized successfully.
	-----------------------------------------------------------------------------
	```
- Check environment was set up correctly.
	```
	C:\>darknet_no_gpu
	usage: darknet_no_gpu <function>
	```
