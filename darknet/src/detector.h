#pragma once
#include "network.h"

__declspec(dllexport) int yolo_detector_v3(const network net, const image pImg, float thresh, int *numDST, float **boxout);