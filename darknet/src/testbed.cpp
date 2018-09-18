#include <iostream>
#include <opencv2/opencv.hpp>
extern "C" {
#include "yolo_v3_dll.h"
}
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#pragma comment(lib,"../darknet/Release/darknet_no_gpu.lib")
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")

typedef struct {
	int num;
	float **Bbox;
}BboxOut;

image _make_empty_image(int w, int h, int c);
image _make_image(int w, int h, int c);
image _ipl_to_image(unsigned char *data, int h, int w, int c, int step);
void _rgbgr_image(image im);
void _free_image(image m);
int main()
{
	while(1)
	{ 
	BboxOut bbox_out = { 0 };
	float thresh = 0.25;
	char *input = "342_1_1_201801.jpg";
	char *cfgfile = "yolov3-voc.cfg";
	char *weightfile = "yolov3-voc_18300.weights";
	network net = parse_network_cfg_custom(cfgfile, 1); // set batch=1
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	//set_batch_network(&net, 1);
	fuse_conv_batchnorm(net);
	IplImage *src = cvLoadImage(input);
	unsigned char *data = (unsigned char *)src->imageData;
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	int step = src->widthStep;
	image out=_ipl_to_image(data,h,w,c,step);
	if (out.c > 1)
		_rgbgr_image(out);
	bbox_out.Bbox = (float**)calloc(100, sizeof(float*));
	for (int i = 0; i < 100; i++)bbox_out.Bbox[i] = (float*)calloc(6, sizeof(float));

	yolo_detector_v3(net, out, thresh,&bbox_out.num,bbox_out.Bbox);
	printf("%d\n", bbox_out.num);
	for (int i = 0; i < bbox_out.num; i++)
	{
		cvRectangle(src,cvPoint(bbox_out.Bbox[i][0], bbox_out.Bbox[i][2]), cvPoint(bbox_out.Bbox[i][1], bbox_out.Bbox[i][3]),cvScalar(255,0,0));
		printf("%f %f %f %f\n",bbox_out.Bbox[i][0], bbox_out.Bbox[i][1], bbox_out.Bbox[i][2], bbox_out.Bbox[i][3]);
	}
	cvShowImage("result",src);
	cvWaitKey();
	_free_image(out);
	free_network(net);
	cvReleaseImage(&src);
	for (int i = 0; i < 100; i++)
	{
		free(bbox_out.Bbox[i]);
		bbox_out.Bbox[i] = NULL;
	}
	free(bbox_out.Bbox);
	bbox_out.Bbox = NULL;
	printf("\n");
	}
	return 0;
}

image _make_empty_image(int w, int h, int c)
{
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}

image _make_image(int w, int h, int c)
{
	image out = _make_empty_image(w, h, c);
	out.data = (float*)calloc(h*w*c, sizeof(float));
	return out;
}
image _ipl_to_image(unsigned char *data, int h, int w, int c, int step)
{

	image out = _make_image(w, h, c);
	int i, j, k, count = 0;;

	for (k = 0; k < c; ++k) {
		for (i = 0; i < h; ++i) {
			for (j = 0; j < w; ++j) {
				out.data[count++] = data[i*step + j*c + k] / 255.;
			}
		}
	}
	return out;
}
void _rgbgr_image(image im)
{
	int i;
	for (i = 0; i < im.w*im.h; ++i) {
		float swap = im.data[i];
		im.data[i] = im.data[i + im.w*im.h * 2];
		im.data[i + im.w*im.h * 2] = swap;
	}
}
void _free_image(image m)
{
	if (m.data) {
		free(m.data);
	}
}