#define CV_NO_BACKWARD_COMPATIBILITY  

#include "cv.h"  
#include "highgui.h"  
#include "io.h"
#include "vector"
#include "string"
#include <iostream>  
#include <cstdio>  
#include <string.h>  
#include <fstream>


#ifdef _EiC  
#define WIN32  
#endif  

using namespace std;
using namespace cv;

void detectAndDraw(Mat& img,
	CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
	double scale, int picid);

int compressPCA(const Mat& pcaset, int maxComponents,
	const Mat& testset, Mat& compressed);

String cascadeName = "haarcascade_frontalface_alt.xml";
String nestedCascadeName = "haarcascade_eye_tree_eyeglasses.xml";

int WINDOWID = 1;

int main(int argc, const char** argv)
{
	Mat frame, frameCopy, image;
	const String scaleOpt = "--scale=";
	size_t scaleOptLen = scaleOpt.length();
	const String cascadeOpt = "--cascade=";
	size_t cascadeOptLen = cascadeOpt.length();
	const String nestedCascadeOpt = "--nested-cascade";
	size_t nestedCascadeOptLen = nestedCascadeOpt.length();
	String inputName;

	CascadeClassifier cascade, nestedCascade;
	double scale = 1;

	argv[1] = "taylor/11.jpg";

	for (int i = 1; i < argc; i++)
	{
		if (cascadeOpt.compare(0, cascadeOptLen, argv[i], cascadeOptLen) == 0)
			cascadeName.assign(argv[i] + cascadeOptLen);
		else if (nestedCascadeOpt.compare(0, nestedCascadeOptLen, argv[i], nestedCascadeOptLen) == 0)
		{
			if (argv[i][nestedCascadeOpt.length()] == '=')
				nestedCascadeName.assign(argv[i] + nestedCascadeOpt.length() + 1);
			if (!nestedCascade.load(nestedCascadeName))
				cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
		}
		else if (scaleOpt.compare(0, scaleOptLen, argv[i], scaleOptLen) == 0)
		{
			if (!sscanf(argv[i] + scaleOpt.length(), "%lf", &scale) || scale < 1)
				scale = 1;
		}
		else if (argv[i][0] == '-')
		{
			cerr << "WARNING: Unknown option %s" << argv[i] << endl;
		}
		else
			inputName.assign(argv[i]);
	}

	if (!cascade.load(cascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		cerr << "Usage: facedetect [--cascade=\"<cascade_path>\"]\n"
			"   [--nested-cascade[=\"nested_cascade_path\"]]\n"
			"   [--scale[=<image scale>\n"
			"   [filename|camera_index]\n";
		return -1;
	}

	inputName = "pic/10.jpg";
	image = imread(inputName, 1);

	if (!image.empty())
	{
		detectAndDraw(image, cascade, nestedCascade, scale, 0);
		waitKey(0);
	}
	else if (!inputName.empty())
	{
		FILE* f = fopen(inputName.c_str(), "rt");
		if (f)
		{
			char buf[1000 + 1];
			int picid = 1;
			printf("################################################################\n");
			printf("\n");
			while (fgets(buf, 1000, f))
			{

				int len = (int)strlen(buf), c;
				while (len > 0 && isspace(buf[len - 1]))
					len--;
				buf[len] = '\0';
				cout << "file " << buf << endl;
				image = imread(buf, 1);
				if (!image.empty())
				{
					detectAndDraw(image, cascade, nestedCascade, scale, picid);
					picid++;
					printf("\n");
					printf("################################################################\n");
					printf("\n");
					printf("Picture ID: %d \n", picid);
				}

			}
			fclose(f);
			waitKey(0);
		}
	}

	cin.get();

	return 0;
}

void getFiles(string path, string exd, vector<string>& files)
{
	//Handle of the picture
	long   hFile = 0;
	//file information
	struct _finddata_t fileinfo;
	string pathName, exdName;

	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "\\*." + exd;
	}
	else
	{
		exdName = "\\*";
	}

	if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
	{
		do
		{
			//If there are sub files under a file, recursivly scan all the files.
			//If not, add to the list.
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(pathName.assign(path).append("\\").append(fileinfo.name), exd, files);
			}
			else
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


void detectAndDraw(Mat& img,
	CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
	double scale, int picid)
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	const static Scalar colors[] = { CV_RGB(0, 0, 255),
		CV_RGB(0, 128, 255),
		CV_RGB(0, 255, 255),
		CV_RGB(0, 255, 0),
		CV_RGB(255, 128, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(255, 0, 0),
		CV_RGB(255, 0, 255) };
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

	cvtColor(img, gray, CV_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	t = (double)cvGetTickCount();
	cascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		| CV_HAAR_SCALE_IMAGE
		,
		Size(30, 30));
	t = (double)cvGetTickCount() - t;
	printf("The time that used for detection£º %g ms\n", t / ((double)cvGetTickFrequency()*1000.));


	int index = 1;
	char facenum[100];
	char filenum[100];
	char filename[255];
	//pca algorithm starts here.  

	stringstream ss;
	char* basic_path = "D:\\EC500\\projecttest1\\OpencvTest\\FR3\\FR3\\facelib\\";
	ss << basic_path;
	ss << "\\";

	
	char buffer[256];
	ifstream in("D:\\EC500\\projecttest1\\OpencvTest\\FR3\\FR3\\facelib\\liblist.txt");
	if (!in.is_open())
	{
		cout << "Error opening file"; exit(1);
	}
	while (!in.eof())
	{
		in.getline(buffer, 1000);
		char * filePath = "D:\\EC500\\projecttest1\\OpencvTest\\FR3\\FR3\\facelib\\";
		stringstream ss;
		ss << filePath << buffer;
		cout << ss.str();
	
	}

	
	vector<string> files;

	//get all the jpg files under this path
	//getFiles(filePath, "jpg", files);
	

	int size = files.size();
	int Arrsize = size * 10000;
	Mat *p = new Mat[size];
	int start = 0;
	float *dataArr = new float[Arrsize];
	for (int i = 0; i < size; i++)
	{
		p[i] = cv::imread(files[i].c_str(), 0);
		resize(p[i], p[i], Size(100, 100));
		for (int x = 0; x<p[i].rows; x++)
		{
			for (int y = 0; y<p[i].cols; y++)
			{
				dataArr[start] = (float)p[i].at<uchar>(x, y);
				start++;
			}
		}
		//cout << files[i].c_str() << endl;
	}

	Mat compressed, UnknowFaceMat;
	/*Mat p1, p2, p3, p4, compressed, UnknowFaceMat;
	
	p1 = cv::imread("positive/1.jpg", 0);
	p2 = cv::imread("positive/2.jpg", 0);
	p3 = cv::imread("positive/3.jpg", 0);
	p4 = cv::imread("positive/4.jpg", 0);
	resize(p1, p1, Size(100, 100));
	resize(p2, p2, Size(100, 100));
	resize(p3, p3, Size(100, 100));
	resize(p4, p4, Size(100, 100));
	float dataArr[40000];
	int start = 0;
	//float dataArr[Arrsize];
	

	
	for (int x = 0; x<p1.rows; x++)
	{
		for (int j = 0; j<p1.cols; j++)
		{
			dataArr[start] = (float)p1.at<uchar>(i, j);
			start++;
		}
	}
	for (int i = 0; i<p2.rows; i++)
	{
		for (int j = 0; j<p2.cols; j++)
		{
			dataArr[start] = (float)p2.at<uchar>(i, j);
			//system.out.println();
			start++;
		}
	}
	for (int i = 0; i<p3.rows; i++)
	{
		for (int j = 0; j<p3.cols; j++)
		{
			dataArr[start] = (float)p3.at<uchar>(i, j);
			start++;
		}
	}
	for (int i = 0; i<p4.rows; i++)
	{
		for (int j = 0; j<p4.cols; j++)
		{
			dataArr[start] = (float)p4.at<uchar>(i, j);
			start++;
		}
	}
	int pos = 0;
	for (int i = 0; i < 40000; i++)
	{
		cout << dataArr[pos] << ", ";
		//printf(dataArr[pos]);
		pos++;
	}
	Mat positiveMat(4, 10000, CV_32FC1, dataArr);*/
	Mat positiveMat(size, 10000, CV_32FC1, dataArr);
	//for second person lib
	char * filePath1 = "D:\\EC500\\projecttest1\\OpencvTest\\FR3\\FR3\\facelib\\positive";
	vector<string> files1;

	//get all the jpg files under this path
	getFiles(filePath1, "jpg", files1);


	int size1 = files1.size();
	int Arrsize1 = size1 * 10000;
	Mat *p1 = new Mat[size];
	int start1 = 0;
	float *dataArr1 = new float[Arrsize1];
	for (int i = 0; i < size1; i++)
	{
		p1[i] = cv::imread(files1[i].c_str(), 0);
		resize(p1[i], p1[i], Size(100, 100));
		for (int x = 0; x<p1[i].rows; x++)
		{
			for (int y = 0; y<p1[i].cols; y++)
			{
				dataArr1[start1] = (float)p1[i].at<uchar>(x, y);
				start1++;
			}
		}
		//cout << files[i].c_str() << endl;
	}

	Mat compressed1;
	Mat positiveMat1(size1, 10000, CV_32FC1, dataArr1);
	//pca algorithm ends 

	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		Mat smallImgROI, tempMat;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);
		// circle( img, center, radius, color, 3, 8, 0 );  

		smallImgROI = smallImg(*r);

		tempMat = smallImg(*r);

		_itoa(picid, filenum, 10);
		_itoa(index, facenum, 10);
		strcpy(filename, "faces/");
		strcat(filename, filenum);
		strcat(filename, "_");
		strcat(filename, facenum);
		strcat(filename, ".jpg");

		cv::Mat graysmallface(Size(100, 100), CV_8UC1);
		cv::resize(tempMat, graysmallface, graysmallface.size(), 0, 0);


		cv::imwrite(filename, graysmallface);

		printf("Detected faces, save into: %s\n", filename);

		cv::imshow("face", graysmallface);
		waitKey();
		index++;

		UnknowFaceMat = graysmallface;
		float UnknowDataArr[10000];
		start = 0;
		for (int i = 0; i<UnknowFaceMat.rows; i++)
		{
			for (int j = 0; j<UnknowFaceMat.cols; j++)
			{
				UnknowDataArr[start] = (float)UnknowFaceMat.at<uchar>(i, j);
				start++;
			}
		}

		Mat negativeMat(1, 10000, CV_32FC1, UnknowDataArr);

		int result = compressPCA(positiveMat, 3, negativeMat, compressed);
		
		if (result == -1)
		{
			cout << "Can't find a match from the face library." << endl;
			return;
		}

		//--use pca end---  


		if (result == 1)
		{
			printf("The person in the picture is Taylor£¡\n press anykeys to continue \n");

			center.x = cvRound((r->x + r->width*0.5)*scale);
			center.y = cvRound((r->y + r->height*0.5)*scale);
			radius = cvRound((r->width + r->height)*0.25*scale);
			circle(img, center, radius, color, 2, 8, 0);

			char windowname[255];
			char wnum[10];
			_itoa(WINDOWID, wnum, 10);
			strcpy(windowname, "Match_");
			strcat(windowname, wnum);

			cv::Mat tempimg(Size(800, 600), CV_8UC3);
			cv::resize(img, tempimg, tempimg.size(), 0, 0);

			cv::imshow(windowname, img);
			WINDOWID++;
			waitKey(0);
		}
		int result1 = compressPCA(positiveMat1, 3, negativeMat, compressed1);
		if (result1 == -1)
		{
			cout << "Can't find a match from the face library." << endl;
			return;
		}

		//--use pca end-0--  


		if (result1 == 1)
		{
			printf("The person in picture is Allen£¡\n press anykeys to continue \n");

			center.x = cvRound((r->x + r->width*0.5)*scale);
			center.y = cvRound((r->y + r->height*0.5)*scale);
			radius = cvRound((r->width + r->height)*0.25*scale);
			circle(img, center, radius, color, 2, 8, 0);

			char windowname[255];
			char wnum[10];
			_itoa(WINDOWID, wnum, 10);
			strcpy(windowname, "Match_");
			strcat(windowname, wnum);

			cv::Mat tempimg(Size(800, 600), CV_8UC3);
			cv::resize(img, tempimg, tempimg.size(), 0, 0);

			cv::imshow(windowname, img);
			WINDOWID++;
			waitKey(0);
		}
	}
}

int compressPCA(const Mat& pcaset, int maxComponents,
	const Mat& testset, Mat& compressed)
{
	PCA pca(pcaset, // pass the data  
		Mat(), // we do not have a pre-computed mean vector,  
		// so let the PCA engine to compute it  
		CV_PCA_DATA_AS_ROW, // indicate that the vectors  
		// are stored as matrix rows  
		// (use CV_PCA_DATA_AS_COL if the vectors are  
		// the matrix columns)  
		maxComponents // specify, how many principal components to retain  
		);
	// if there is no test data, just return the computed basis, ready-to-use  
	//if( !testset.data )  1
	//return pca;  
	CV_Assert(testset.cols == pcaset.cols);
	compressed.create(testset.rows, maxComponents, testset.type());
	Mat reconstructed;
	cout << "testset.rows = " << testset.rows << endl;
	for (int i = 0; i < testset.rows; i++)
	{
		Mat vec = testset.row(i), coeffs = compressed.row(i);
		// compress the vector, the result will be stored  
		// in the i-th row of the output matrix  
		pca.project(vec, coeffs);
		// and then reconstruct it  
		pca.backProject(coeffs, reconstructed);
		// and measure the error  
		double norm_result = norm(vec, reconstructed, NORM_L2);
		printf("%d. result£º %g\n", i + 1, norm_result);

		if (norm_result > 7000)
		{
			return -1;
		}

	}
	return 1;
}