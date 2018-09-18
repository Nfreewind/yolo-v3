#ifndef PARSER_H
#define PARSER_H
#include "network.h"
#include "yolo_v3_dll.h"

network parse_network_cfg(char *filename);
//__declspec(dllexport) network parse_network_cfg_custom(char *filename, int batch);
void save_network(network net, char *filename);
void save_weights(network net, char *filename);
void save_weights_upto(network net, char *filename, int cutoff);
void save_weights_double(network net, char *filename);
//__declspec(dllexport) void load_weights(network *net, char *filename);
void load_weights_upto(network *net, char *filename, int cutoff);

#endif
