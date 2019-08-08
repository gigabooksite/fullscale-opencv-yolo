@echo off

set ROOT=%~dp0
call :GetFullPath "%ROOT%\.." ROOT

set "INTEL_OPENVINO_DIR=C:\Program Files (x86)\IntelSWTools\openvino_2019.1.148"
set "INTEL_CVSDK_DIR=%INTEL_OPENVINO_DIR%"

where /q libmmd.dll || echo Warning: libmmd.dll couldn't be found in %%PATH%%. Please check if the redistributable package for Intel(R) C++ Compiler is installed and the library path is added to the PATH environment variable. System reboot can be required to update the system environment.

:: OpenCV
if exist "%INTEL_OPENVINO_DIR%\opencv\setupvars.bat" (
call "%INTEL_OPENVINO_DIR%\opencv\setupvars.bat"
) else (
set "OpenCV_DIR=%INTEL_OPENVINO_DIR%\opencv\x64\vc14\lib"
set "PATH=%INTEL_OPENVINO_DIR%\opencv\x64\vc14\bin;%PATH%"
)

:: OpenVX
set "OPENVX_FOLDER=%INTEL_OPENVINO_DIR%\openvx"
set "PATH=%INTEL_OPENVINO_DIR%\openvx\bin;%PATH%"

:: Inference Engine
set "InferenceEngine_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\share"
set "HDDL_INSTALL_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\hddl"
set "PATH=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Release;%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Debug;%HDDL_INSTALL_DIR%\bin;%PATH%"
if exist "%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\arch_descriptions" (
set "ARCH_ROOT_DIR=%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\arch_descriptions"
)

:: Darknet
set "PATH=%ROOT%\build\darknet\x64;%PATH%"
where /q darknet.exe || echo Warning: darknet.exe couldn't be found.
if %errorlevel% NEQ 0 goto ExitWithError

echo 

:End
set exitCode=0
echo ---------------------------------------------------------------------------
echo -- SUCCESS. Darknet environment initialized successfully.
echo ---------------------------------------------------------------------------
goto Term

:ExitWithError
set exitCode=1
echo.
echo ---------------------------------------------------------------------------
echo -- ERROR(s) encountered. Darknet environment NOT initialized. 
echo ---------------------------------------------------------------------------
goto Term

:Term
exit /B %exitCode%

:GetFullPath
SET %2=%~f1

GOTO :EOF