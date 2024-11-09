/*
* This sample demonstrates the use of the function
* findTransformECC that implements the image alignment ECC algorithm
*
*
* The demo loads an image (defaults to fruits.jpg) and it artificially creates
* a template image based on the given motion type. When two images are given,
* the first image is the input image and the second one defines the template image.
* In the latter case, you can also parse the warp's initialization.
*
* Input and output warp files consist of the raw warp (transform) elements
*
* Authors: G. Evangelidis, INRIA, Grenoble, France
*          M. Asbach, Fraunhofer IAIS, St. Augustin, Germany
*/
#include "ecc_cuda.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <stdio.h>
#include <string>
#include <time.h>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;
static void help(const char** argv);
static int readWarp(string iFilename, Mat& warp, int motionType);
static int saveWarp(string fileName, const Mat& warp, int motionType);
static void draw_warped_roi(Mat& image, const int width, const int height, Mat& W);
#define HOMO_VECTOR(H, x, y)\
	H.at<float>(0,0) = (float)(x);\
	H.at<float>(1,0) = (float)(y);\
	H.at<float>(2,0) = 1.;
#define GET_HOMO_VALUES(X, x, y)\
	(x) = static_cast<float> (X.at<float>(0,0)/X.at<float>(2,0));\
	(y) = static_cast<float> (X.at<float>(1,0)/X.at<float>(2,0));
const std::string keys =
"{@inputImage    | imgs/img1.jpg    | input image filename }"
"{@templateImage | imgs/img2.jpg       | template image filename (optional)}"
"{@inputWarp     |               | input warp (matrix) filename (optional)}"
"{n numOfIter    | 50            | ECC's iterations }"
"{e epsilon      | 0.0001        | ECC's convergence epsilon }"
"{o outputWarp   | out/outWarp.ecc   | output warp (matrix) filename }"
"{m motionType   | affine        | type of motion (translation, euclidean, affine, homography) }"
"{v verbose      | 1             | display initial and final images }"
"{w warpedImfile | out/warpedECC.png | warped input image }"
"{h help | | print help message }"
;
static void help(const char** argv)
{
	cout << "\nThis file demonstrates the use of the ECC image alignment algorithm. When one image"
		" is given, the template image is artificially formed by a random warp. When both images"
		" are given, the initialization of the warp by command line parsing is possible. "
		"If inputWarp is missing, the identity transformation initializes the algorithm. \n" << endl;
	cout << "\nUsage example (one image): \n"
		<< argv[0]
		<< " fruits.jpg -o=outWarp.ecc "
		"-m=euclidean -e=1e-6 -N=70 -v=1 \n" << endl;
	cout << "\nUsage example (two images with initialization): \n"
		<< argv[0]
		<< " yourInput.png yourTemplate.png "
		"yourInitialWarp.ecc -o=outWarp.ecc -m=homography -e=1e-6 -N=70 -v=1 -w=yourFinalImage.png \n" << endl;
}
static int readWarp(string iFilename, Mat& warp, int motionType) {
	// it reads from file a specific number of raw values:
	// 9 values for homography, 6 otherwise
	CV_Assert(warp.type() == CV_32FC1);
	int numOfElements;
	if (motionType == MOTION_HOMOGRAPHY)
		numOfElements = 9;
	else
		numOfElements = 6;
	int i;
	int ret_value;
	ifstream myfile(iFilename.c_str());
	if (myfile.is_open()) {
		float* matPtr = warp.ptr<float>(0);
		for (i = 0; i<numOfElements; i++) {
			myfile >> matPtr[i];
		}
		ret_value = 1;
	}
	else {
		cout << "Unable to open file " << iFilename.c_str() << endl;
		ret_value = 0;
	}
	return ret_value;
}
static int saveWarp(string fileName, const Mat& warp, int motionType)
{
	// it saves the raw matrix elements in a file
	CV_Assert(warp.type() == CV_32FC1);
	const float* matPtr = warp.ptr<float>(0);
	int ret_value;
	ofstream outfile(fileName.c_str());
	if (!outfile) {
		cerr << "error in saving "
			<< "Couldn't open file '" << fileName.c_str() << "'!" << endl;
		ret_value = 0;
	}
	else {//save the warp's elements
		outfile << matPtr[0] << " " << matPtr[1] << " " << matPtr[2] << endl;
		outfile << matPtr[3] << " " << matPtr[4] << " " << matPtr[5] << endl;
		if (motionType == MOTION_HOMOGRAPHY) {
			outfile << matPtr[6] << " " << matPtr[7] << " " << matPtr[8] << endl;
		}
		ret_value = 1;
	}
	return ret_value;
}
static void draw_warped_roi(Mat& image, const int width, const int height, Mat& W)
{
	Point2f top_left, top_right, bottom_left, bottom_right;
	Mat  H = Mat(3, 1, CV_32F);
	Mat  U = Mat(3, 1, CV_32F);
	Mat warp_mat = Mat::eye(3, 3, CV_32F);
	for (int y = 0; y < W.rows; y++)
		for (int x = 0; x < W.cols; x++)
			warp_mat.at<float>(y, x) = W.at<float>(y, x);
	//warp the corners of rectangle
	// top-left
	HOMO_VECTOR(H, 1, 1);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, top_left.x, top_left.y);
	// top-right
	HOMO_VECTOR(H, width, 1);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, top_right.x, top_right.y);
	// bottom-left
	HOMO_VECTOR(H, 1, height);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, bottom_left.x, bottom_left.y);
	// bottom-right
	HOMO_VECTOR(H, width, height);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, bottom_right.x, bottom_right.y);
	// draw the warped perimeter
	line(image, top_left, top_right, Scalar(255));
	line(image, top_right, bottom_right, Scalar(255));
	line(image, bottom_right, bottom_left, Scalar(255));
	line(image, bottom_left, top_left, Scalar(255));
}

void init_params(const CommandLineParser& parser, string& imgFile, string& tempImgFile, string& inWarpFile, string& warpType, string& finalWarp,
				string& warpedImFile, int& number_of_iterations, int& verbose, double& termination_eps) {
	imgFile = parser.get<string>(0);
	tempImgFile = parser.get<string>(1);
	inWarpFile = parser.get<string>(2);
	number_of_iterations = parser.get<int>("n");
	termination_eps = parser.get<double>("e");
	warpType = parser.get<string>("m");
	verbose = parser.get<int>("v");
	finalWarp = parser.get<string>("o");
	warpedImFile = parser.get<string>("w");
}

int main(const int argc, const char * argv[])
{
	const CommandLineParser parser(argc, argv, keys);
	//parser.about("ECC demo");
	parser.printMessage();
	help(argv);

	string imgFile, tempImgFile, inWarpFile, warpType, finalWarp, warpedImFile;
	int number_of_iterations, verbose;
	double termination_eps;
	init_params(parser, imgFile, tempImgFile, inWarpFile, warpType, finalWarp, warpedImFile, number_of_iterations, verbose, termination_eps);


	if (!parser.check())
	{
		parser.printErrors();
		return -1;
	}
	if (!(warpType == "translation" || warpType == "euclidean"
		|| warpType == "affine" || warpType == "homography"))
	{
		cerr << "Invalid motion transformation" << endl;
		return -1;
	}
	int mode_temp;
	if (warpType == "translation")
		mode_temp = MOTION_TRANSLATION;
	else if (warpType == "euclidean")
		mode_temp = MOTION_EUCLIDEAN;
	else if (warpType == "affine")
		mode_temp = MOTION_AFFINE;
	else
		mode_temp = MOTION_HOMOGRAPHY;
	//Mat inputImage = imread(samples::findFile(imgFile), IMREAD_GRAYSCALE);

	std::string basePath = "C:/Users/hadar/Documents/gitRepos/Hadar/VideoMotionDetection/";

	imgFile			= basePath + imgFile;
	finalWarp		= basePath + finalWarp;
	warpedImFile	= basePath + warpedImFile;

	Mat inputImage = imread(imgFile, IMREAD_GRAYSCALE);
	Mat target_image, template_image;

	inputImage.copyTo(target_image);
	tempImgFile = basePath + tempImgFile;
	template_image = imread(tempImgFile, IMREAD_GRAYSCALE);

	const int warp_mode = mode_temp;
	// initialize or load the warp matrix
	Mat warp_matrix;
	if (warpType == "homography")
		warp_matrix = Mat::eye(3, 3, CV_32F);
	else
		warp_matrix = Mat::eye(2, 3, CV_32F);

	// Set the path to the video file
	string videoPath = "C:/Users/hadar/Documents/database/videos/drones/drone_pov/aerial_town_view_1.mp4";

	// Open the video file
	VideoCapture cap(videoPath);

	// Check if video opened successfully
	if (!cap.isOpened()) {
		cerr << "Error: Could not open video file." << endl;
		return -1;
	}

	// Get the frame rate of the video
	double fps = cap.get(CAP_PROP_FPS);
	int delay = static_cast<int>(1000 / fps);  // Delay between frames in ms

	// Loop to read and display frames
	while (true) {
		Mat frame;

		// Capture frame-by-frame
		cap >> frame;

		// Check if the frame is empty, which indicates the end of the video
		if (frame.empty()) {
			cout << "End of video reached." << endl;
			break;
		}

		// Display the frame
		cv::imshow("Video Reader", frame);

		// Wait for 'q' key to quit or proceed to the next frame
		if (cv::waitKey(delay) == 'q') {
			break;
		}
	}

	//double cc = findTransformECCGpu(template_image, target_image, warp_matrix, warp_mode, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,	number_of_iterations, termination_eps), gaussian_size);



	// done
	return 0;
}
