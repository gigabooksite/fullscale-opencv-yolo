#include "TeamClassifier.h"

using namespace TeamClassify;

#define DO_GROUND_TRUTH 0 // Set to 1 to output ground truth info to console
#define SHOW_PROFILER_STATS 0 // Set to 1 to output profiler stats to console

TeamClassifier::TeamClassifier() :
	frameIdx(0)
{

}

TeamClassifier::~TeamClassifier()
{
	
}

/**
 * @brief Gets the distance of two centers.
 * @param center1 first center
 * @param center2 second center
 * @return the distance
 */
inline static double GetDist(cv::Scalar center1, cv::Scalar center2)
{
	return cv::sqrt(cv::pow(center1[0] - center2[0], 2) + cv::pow(center1[1] - center2[1], 2) + cv::pow(center1[2] - center2[2], 2));
}

int TeamClassifier::ProcessFrame(FrameProcParams* params)
{	
	frameIdx++;
	int code = CODE_ERROR;
	FrameData frameCtxt;

	for (;;)
	{
		if (nullptr == params || params->frame.empty() || params->classNames.empty())
		{
			code = CODE_BAD_ARG;
			break;
		}		

		int64 procFrameRef = cv::getTickCount();
		// Get info for current frame.		
		for (size_t i = 0; i < params->indices.size(); ++i)
		{
			BoxProps boxProps;
			int idx = params->indices[i];
			boxProps.classId = params->classIds[idx];
			boxProps.boxNms = params->boxesNms[idx];
			boxProps.boxNmsIdx = idx;
			boxProps.confidence = params->confidence[idx];
			
			if (params->classNames[boxProps.classId].compare("player") == 0)
			{
				// Extract player's general base color.
				frameCtxt.playerBoxPropIndices.push_back((int)i);
				boxProps.isPlayer = true;
				int64 ref = cv::getTickCount();
				GetPlayerBaseColor(params->frame, boxProps.boxNms, boxProps.genBaseColorRoi, boxProps.genBaseColor);
				Common::GetPerf(this->profProcGetPlayerBaseColor, ref);		
			}

			frameCtxt.boxProps.push_back(boxProps);			
		}

		// Create teams (if needed) then add players to their respective teams.
		if (frameCtxt.playerBoxPropIndices.size() > 1)
		{
			// Split players, in current frame, into two teams based on color.
			cv::Mat playerColors = cv::Mat::zeros((int)frameCtxt.playerBoxPropIndices.size(), 3, CV_8UC1);
			uchar* dest = playerColors.data;
			for (size_t i = 0; i < frameCtxt.boxProps.size(); ++i)
			{
				if (frameCtxt.boxProps[i].isPlayer)
				{
					*dest++ = (uchar)frameCtxt.boxProps[i].genBaseColor[0]; // B
					*dest++ = (uchar)frameCtxt.boxProps[i].genBaseColor[1]; // G
					*dest++ = (uchar)frameCtxt.boxProps[i].genBaseColor[2]; // R					
				}
			}			
			cv::Mat labels;
			cv::Mat centers;
			const int clusterNum = 2;
			cv::TermCriteria criteria{ cv::TermCriteria::COUNT, 10, 1 };
			cv::Mat playerColors_f32; playerColors.convertTo(playerColors_f32, CV_32FC1, 1.0 / 255.0);
			cv::kmeans(playerColors_f32, clusterNum, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);

			// If centers are too close then assume the players are from the same team.
			cv::Mat centers_u8; centers.convertTo(centers_u8, CV_8UC1, 255.0);
			cv::Scalar center1(centers_u8.at<uchar>(0), centers_u8.at<uchar>(1), centers_u8.at<uchar>(2));
			cv::Scalar center2(centers_u8.at<uchar>(3), centers_u8.at<uchar>(4), centers_u8.at<uchar>(5));			
			if (GetDist(center1, center2) <= DIST_THRESH)
			{
				// Players are from same team.
				for (int i = 0; i < frameCtxt.playerBoxPropIndices.size(); i++)
				{
					FindAddTeam(frameCtxt.boxProps[frameCtxt.playerBoxPropIndices[i]], center1);
				}
			}
			else
			{				
				// We have players from 2 teams.
				for (int i = 0; i < frameCtxt.playerBoxPropIndices.size(); i++)
				{
					int boxIdx = frameCtxt.playerBoxPropIndices[i];
					// Identify closer center and which will be used if we need to add a new team based on the center's color.
					double dst = GetDist(center1, frameCtxt.boxProps[boxIdx].genBaseColor);
					cv::Scalar center = center1;
					if (dst > GetDist(center2, frameCtxt.boxProps[boxIdx].genBaseColor))
					{
						center = center2;
					}
					FindAddTeam(frameCtxt.boxProps[boxIdx], center);
				}
			}
		}
		else if (1 == frameCtxt.playerBoxPropIndices.size())
		{
			// Handle only 1 player.
			int boxIdx = frameCtxt.playerBoxPropIndices[0];
			FindAddTeam(frameCtxt.boxProps[boxIdx], frameCtxt.boxProps[boxIdx].genBaseColor);
		}
		else
		{
			// TODO: Handle no players
		}

		frameCtxts.push_back(frameCtxt);

#if 0
		// Draw player base.
		cv::Mat frame = params->frame.clone();
		for (int i = 0; i < frameCtxt.playerIndices.size(); i++)
		{			
			BoxProps* props = &frameCtxt.boxProps[frameCtxt.playerIndices[i]];
			cv::Rect box = params->boxes[props->boxIndex];
			cv::RotatedRect rot = cv::RotatedRect(cv::Point(box.x + box.width / 2, box.y + box.height), cv::Size(50, 16), 0);
			cv::ellipse(frame, rot, props->genBaseColor, 20);
			std::cout << box << std::endl;
		}
#endif

		// Save teams for this frame.
		lastTeams[0].props = teamProps[0]; lastTeams[0].playerBoxNmsIndices.clear();
		lastTeams[1].props = teamProps[1]; lastTeams[1].playerBoxNmsIndices.clear();
		for (int i = 0; i < frameCtxt.playerBoxPropIndices.size(); i++)
		{
			BoxProps* props = &frameCtxt.boxProps[frameCtxt.playerBoxPropIndices[i]];
			lastTeams[props->teamIdx].playerBoxNmsIndices.push_back(props->boxNmsIdx);
		}

		Common::GetPerf(this->profProcProcessFrame, procFrameRef);
		code = CODE_OK;
		break; // Mandatory break for one iteration
	}

#if DO_GROUND_TRUTH	
	if (1 == frameIdx)
	{
		// Write header once.
		// <frame idx>, <return code>, <player count>,<reserve>, <P1 box idx>, <P1 team idx>, <P1 x>, <P1 y>, <P1 width>, <P1 height>, ... <Pn box idx>, <Pn team idx>, <Pn x>, <Pn y>, <Pn width>, <Pn height>, 
		std::cout << "\n<frame idx>,<return code>,<player count>,<reserve>,<team idx>,<box idx>,<x>,<y>,<width>,<height>" << std::endl;
	}
	
	// Write rows.
	std::cout << std::dec << frameIdx << "," << code << "," << frameCtxt.playerBoxPropIndices.size() << ",";
	// Sort players left to right related to position in frame.
	std::vector<int> sorted = frameCtxt.playerBoxPropIndices;
	std::sort(sorted.begin(), sorted.end(), [frameCtxt](const int& lhs, const int& rhs)
	{	
		return (frameCtxt.boxProps[lhs].boxNms.x < frameCtxt.boxProps[rhs].boxNms.x);
	});
	for (size_t i = 0; i < sorted.size(); i++)
	{
		BoxProps* props = &frameCtxt.boxProps[sorted[i]]; cv::Rect boxNms = props->boxNms;
		std::cout << "," << props->teamIdx << "," << props->boxNmsIdx << "," << 
			boxNms.x << "," << boxNms.y << "," << boxNms.width << "," << boxNms.height;
	}
	
	std::cout << std::endl;
#endif

#if SHOW_PROFILER_STATS
	std::cout << "\n -- FRAME " << std::dec << std::left << std::setw(3) << frameIdx << " - TeamClassifier::GetPlayerBaseColor:";
	Common::ProfStats* stats = &this->profProcGetPlayerBaseColor;
	stats->averageTime = stats->totalTime / stats->count;
	std::cout << "\n    AVG: " << (stats->averageTime * 1e+6) << " us,  COUNT: " << std::dec << stats->count;
	std::cout << "\n    MIN: " << (stats->minTime * 1e+6) << " us ";
	std::cout << "\n    MAX: " << (stats->maxTime * 1e+6) << " us ";

	std::cout << "\n -- FRAME " << std::dec << std::left << std::setw(3) << frameIdx << " - TeamClassifier::ProcessFrame:";
	stats = &this->profProcProcessFrame;
	stats->averageTime = stats->totalTime / stats->count;
	std::cout << "\n    AVG: " << (stats->averageTime * 1e+6) << " us,  COUNT: " << std::dec << stats->count;
	std::cout << "\n    MIN: " << (stats->minTime * 1e+6) << " us ";
	std::cout << "\n    MAX: " << (stats->maxTime * 1e+6) << " us ";
#endif
	
	return code;
}

void TeamClassifier::GetFrameProcResult(FrameData& result)
{
	result = frameCtxts.back();
}

void TeamClassifier::GetTeams(Team& teamA, Team& teamB)
{
	teamA = lastTeams[0];
	teamB = lastTeams[1];
}

bool TeamClassifier::GetPlayerBaseColor(const cv::Mat& frame, const cv::Rect playerRect, cv::Rect& roi, cv::Scalar& bgr)
{
	bool success = false;

	if (!frame.empty() && playerRect.width > 0 && playerRect.height > 0)
	{
		// Focus on smaller portion of the blob - the jersey ROI.
		int boxRectNewWidth = playerRect.width / 4;
		int boxRectNewHeight = playerRect.height - (playerRect.height / 4);
		roi = cv::Rect(
			(playerRect.x + playerRect.width / 2) - boxRectNewWidth / 2,
			(playerRect.y + playerRect.height / 2) - boxRectNewHeight / 2,
			boxRectNewWidth,
			boxRectNewHeight);
		cv::Mat jerseyRoi = frame(roi); 
		
		// Get HSV.
		cv::Mat roiHsv;
		cv::cvtColor(jerseyRoi, roiHsv, cv::COLOR_BGR2HSV);
		cv::Mat h, s, v;
		std::vector<cv::Mat> ch;
		cv::split(roiHsv, ch);
		h = ch[0]; s = ch[1]; v = ch[2];

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

#if 0
		// Show updated image using single color.
		cv::Mat kinImgOut;
		Common::DrawKmeansImage(kinImgOut, labels, centers, kinImg.rows, kinImg.cols);
#endif	

		success = true;
	}

	return success;
}

int TeamClassifier::FindTeamIndex(cv::Scalar baseColor) const
{
	double dst = 255.0;
	int teamIdx = -2; // Cannot associate to known teams.

	if (teamProps[0].isValid && teamProps[1].isValid)
	{
		// With both teams already identified we just find the closest.
		dst = GetDist(baseColor, teamProps[0].genBaseColor);
		teamIdx = 0;
		double dst2 = GetDist(baseColor, teamProps[1].genBaseColor);
		if (dst > dst2)
		{
			dst = dst2;
			teamIdx = 1;
		}
		dst = 1.0; // Override to accept closest.
	}
	else if (teamProps[0].isValid)
	{
		dst = GetDist(baseColor, teamProps[0].genBaseColor);
		teamIdx = 0;
	}
	else if (teamProps[1].isValid)
	{
		dst = GetDist(baseColor, teamProps[1].genBaseColor);
		teamIdx = 1;
	}
	else
	{
		teamIdx = -1; // No valid teams
	}

	if (-1 != teamIdx && dst > DIST_THRESH)
	{
		teamIdx = -2; // Cannot associate to with any known teams.
	}
	return teamIdx;
}

int TeamClassifier::AddNewTeam(cv::Scalar baseColor)
{
	int teamIdx = -1;
	if (false == teamProps[0].isValid)
	{
		teamIdx = 0;	
	}
	else if (false == teamProps[1].isValid)
	{
		teamIdx = 1;
	}
	if (teamIdx >= 0)
	{
		teamProps[teamIdx].genBaseColor = baseColor;
		teamProps[teamIdx].color = baseColor;
		teamProps[teamIdx].isValid = true;
	}
	return teamIdx;
}

int TeamClassifier::FindAddTeam(BoxProps& boxProps, cv::Scalar newTeamBaseColor)
{
	int tIdx = FindTeamIndex(newTeamBaseColor);
	if (0 > tIdx)
	{
		tIdx = AddNewTeam(newTeamBaseColor);
	}
	if (tIdx >= 0)
	{
		boxProps.teamIdx = tIdx;
	}
	else
	{
		tIdx = -1;
		assert(true && "Failed to associate player to a team");
	}	
	return tIdx;
}