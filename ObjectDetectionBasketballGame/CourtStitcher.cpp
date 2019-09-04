#include "CourtStitcher.h"

#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/timelapsers.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>

#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

using namespace std;
using namespace cv;
using namespace cv::detail;

CourtStitcher::CourtStitcher() : calibrated(false)
{
}


CourtStitcher::~CourtStitcher()
{
}

void CourtStitcher::calibrate(const std::vector<cv::Mat>& frames)
{
	size_t num_images = frames.size();

	double work_scale = 1, seam_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false;
	double work_megapix = 0.6;

	Ptr<Feature2D> finder;
#ifdef HAVE_OPENCV_XFEATURES2D
	finder = xfeatures2d::SURF::create();
#else
	finder = ORB::create();
#endif

	Mat img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	full_img_sizes.resize(num_images);
	
	double seam_work_aspect = 1;
	double seam_megapix = 0.1;
	float match_conf = 0.3f;
	float conf_thresh = 1.f;
	int i = 0;

	// Find features
	for (cv::Mat frame : frames)
	{
		full_img_sizes[i] = frame.size();

		if (!is_work_scale_set)
		{
			work_scale = min(1.0, sqrt(work_megapix * 1e6 / frame.size().area()));
			is_work_scale_set = true;
		}
		resize(frame, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);

		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / frame.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}

		computeImageFeatures(finder, img, features[i]);
		features[i].img_idx = i;

		resize(frame, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
		images[i] = img.clone();

		++i;
	}
	img.release();

	// Pairwise matching
	vector<MatchesInfo> pairwise_matches;
	Ptr<FeaturesMatcher> matcher;
	matcher = makePtr<BestOf2NearestMatcher>(false, match_conf);

	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();

	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<Mat> img_subset;
	vector<Size> full_img_sizes_subset;
	for (const auto& x : indices)
	{
		img_subset.push_back(images[x]);
		full_img_sizes_subset.push_back(full_img_sizes[x]);
	}
	images = img_subset;
	full_img_sizes = full_img_sizes_subset;

	Ptr<Estimator> estimator;
	estimator = makePtr<HomographyBasedEstimator>();

	if (!(*estimator)(features, pairwise_matches, cameras))
	{
		cout << "Homography estimation failed." << endl;
		return;
	}

	for (auto& cam : cameras)
	{
		Mat R;
		cam.R.convertTo(R, CV_32F);
		cam.R = R;
	}

	Ptr<detail::BundleAdjusterBase> adjuster;
	adjuster = makePtr<detail::BundleAdjusterRay>();

	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	refine_mask(0, 0) = 1;
	refine_mask(0, 1) = 1;
	refine_mask(0, 2) = 1;
	refine_mask(1, 1) = 1;
	refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);
	if (!(*adjuster)(features, pairwise_matches, cameras))
	{
		cout << "Camera parameters adjusting failed.\n";
		return;
	}

	// Find median focal length
	vector<double> focals;
	for (const auto& cam : cameras)
	{
		focals.push_back(cam.focal);
	}

	sort(focals.begin(), focals.end());
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	vector<Mat> rmats;
	for(const auto& cam : cameras)
	{
		rmats.push_back(cam.R.clone());
	}
	waveCorrect(rmats, detail::WAVE_CORRECT_HORIZ);
	i = 0;
	for (auto& cam : cameras)
	{
		cam.R = rmats[i++];
	}

	// Warp images (auxiliary)
	corners.resize(num_images);
	masks_warped.resize(num_images);
	vector<UMat> images_warped(num_images);
	sizes.resize(num_images);
	vector<UMat> masks(num_images);

	// Preapre images masks
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks
	warper_creator = makePtr<cv::SphericalWarper>();
	if (!warper_creator)
	{
		cout << "Can't create the SphericalWarper. \n";
		return;
	}

	warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
	{
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	}

	// Compensate exposure
	compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
	if (dynamic_cast<GainCompensator*>(compensator.get()))
	{
		GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
		gcompensator->setNrFeeds(1);
	}

	if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
	{
		ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
		ccompensator->setNrFeeds(1);
	}

	if (dynamic_cast<BlocksCompensator*>(compensator.get()))
	{
		BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
		bcompensator->setNrFeeds(1);
		bcompensator->setNrGainsFilteringIterations(2);
		bcompensator->setBlockSize(32, 32);
	}

	compensator->feed(corners, images_warped, masks_warped);

	// Find seams
	Ptr<SeamFinder> seam_finder;
	seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);

	seam_finder->find(images_warped_f, corners, masks_warped);

	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	calibrated = true;
}

Mat CourtStitcher::stitch(const std::vector<cv::Mat>& frames)
{
	Mat img, img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Ptr<Blender> blender;
	Ptr<Timelapser> timelapser;
	bool is_compose_scale_set = false;
	double work_scale = 1, compose_scale = 1;
	double compose_work_aspect = 1;
	int i = 0;
	for (Mat frame : frames)
	{
		if (!is_compose_scale_set)
		{
			is_compose_scale_set = true;

			// Compute relative scales
			compose_work_aspect = compose_scale / work_scale;

			// Update warped image scale
			warped_image_scale *= static_cast<float>(compose_work_aspect);
			warper = warper_creator->create(warped_image_scale);

			// Update corners and sizes
			for (int i = 0; i < frames.size(); ++i)
			{
				// Update intrinsics
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;

				// Update corner and size
				Size sz = full_img_sizes[i];
				if (std::abs(compose_scale - 1) > 1e-1)
				{
					sz.width = cvRound(full_img_sizes[i].width * compose_scale);
					sz.height = cvRound(full_img_sizes[i].height * compose_scale);
				}

				Mat K;
				cameras[i].K().convertTo(K, CV_32F);
				Rect roi = warper->warpRoi(sz, K, cameras[i].R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		if (abs(compose_scale - 1) > 1e-1)
			resize(frame, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
		else
			img = frame;
		frame.release();
		Size img_size = img.size();

		Mat K;
		cameras[i].K().convertTo(K, CV_32F);

		// Warp the current image
		warper->warp(img, K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		compensator->apply(i, corners[i], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();

		dilate(masks_warped[i], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
		mask_warped = seam_mask & mask_warped;

		if (!blender)
		{
			blender = Blender::createDefault(Blender::MULTI_BAND, false);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * 5 / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, false);
			else
			{
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
				mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
			}
			blender->prepare(corners, sizes);
		}

		// Blend the current image
		blender->feed(img_warped_s, mask_warped, corners[i]);

		++i;
	}

	Mat result, result_mask;
	blender->blend(result, result_mask);

	return result;
}

bool CourtStitcher::isCalibrated() const
{
	return calibrated;
}