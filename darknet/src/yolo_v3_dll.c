#include "yolo_v3_dll.h"
#include "image.h"
#include "utils.h"

int yolo_detector_v3(const network net, const image pImg, float thresh, int *numDST, float **boxout)
{
	srand(2222222);
	double time;
	int j;
	float nms = .45;    // 0.4F

	int letterbox = 0;
	float hier_thresh = 0.5;
	image sized = resize_image(pImg, net.w, net.h);
	layer l = net.layers[net.n - 1];

	float *X = sized.data;
	time = what_time_is_it_now();
	network_predict(net, X);
	printf(" Predicted in %f seconds.\n", (what_time_is_it_now() - time));
	int nboxes = 0;
	detection *dets = get_network_boxes(&net, pImg.w, pImg.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
	if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
	box_detections_v3(pImg, dets, nboxes, thresh, numDST, boxout);

	free_detections(dets, nboxes);
	free_image(sized);
}