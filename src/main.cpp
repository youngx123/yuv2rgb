/***
 *@Author       : xyoung
 *@Date         : 2023-08-30 09:14:38
 *@LastEditors  : Do not edit
 *@LastEditTime : 2023-10-17 14:11:27
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <unistd.h>
#include "snpe_infer.h"
#include "nlohmann/json.hpp"

typedef struct
{
	std::string model_file;
	std::string input_file;
	bool use_quant;
} Config;

void readConfig(std::string config_file, Config &cfg)
{
	std::ifstream f(config_file.c_str());
	nlohmann::json data = nlohmann::json::parse(f);
	cfg.input_file = data.at("input_file");
	cfg.model_file = data.at("model_file");
	cfg.use_quant = data.at("use_quant");
}

int main(int argc, char **argv)
{
	if (argc < 4 || argc > 6 || argc == 5)
	{
		printf("usage format error, sample as follows : \n");
		printf("./snpe_Infer \t ./dlc_model \t ./test_img_folder \t device number \n");
		exit(0);
	}
	std::string model_name = argv[1];
	std::string test_dir = argv[2];
	int use_quant = std::atoi(argv[3]);

	printf(" model file  : %s\n", model_name.c_str());
	printf(" input  file  : %s\n", test_dir.c_str());
	printf(" use quant : %d\n", use_quant);

	int device = 0;
	switch (use_quant)
	{
	case 0:
		printf("use cpu backend\n");
		device = 0;
		break;
	case 1:
		printf("use GPU backend\n");
		device = 1;
		break;
	case 2:
		printf("use DSP backend\n");
		device = 2;
		break;
	case 3:
		printf("use APU backend\n");
		device = 3;
		break;
	}

	std::unique_ptr<SnpeInfer> detector;

	sleep(5);
	std::vector<cv::String> name_List;
	cv::glob(test_dir, name_List);

	int height = 720;
	int width = 1280;
	int framesize = width * height * 1.5;
	detector.reset(new SnpeInfer(model_name, width, height, {"output"}, device));

	float total_time = 0.0;
	for (int i = 0; i < name_List.size(); i++)
	{
		std::string file = name_List.at(i);

		uint8_t *yuv_data = new uint8_t[framesize];
		FILE *nv21file = fopen(file.c_str(), "rb");
		fread(yuv_data, framesize * sizeof(uint8_t), 1, nv21file);
		fclose(nv21file);

		cv::Mat yuvImg, rgbImg;
		yuvImg.create(int(height * 1.5), width, CV_8UC1);
		memcpy(yuvImg.data, yuv_data, framesize * sizeof(unsigned uint8_t));

		printf("\n");
		if (yuvImg.empty())
		{
			printf("empty data file %s", file.c_str());
		}

		detector->DoInference(yuvImg, 1, true, file);
	}

	return 0;
}