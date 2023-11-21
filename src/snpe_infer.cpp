#include "snpe_infer.h"
#include "createBuffer.h"

#include <iostream>
#include <fstream>

SnpeInfer::SnpeInfer(const std::string dlc_file, int dst_w, int dst_h, std::vector<std::string> output_node, int device)
{
	dlc_path = dlc_file;
	input_w = dst_w;
	input_h = dst_h;
	output_names = output_node;
	device_ = device;

	bool res = SnpeInfer::initDevice(device);
	if (!res)
	{
		printf("init device failed !!!\n");
	}
	SnpeInfer::Init(dlc_path, runtime, output_names);
}

zdl::DlSystem::Runtime_t SnpeInfer::checkRuntime(zdl::DlSystem::Runtime_t runtime)
{
	static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();

	// Print snpe Version number
	printf("SNPE Version: %s \n", Version.asString().c_str());

	if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime))
	{
		printf("Selected runtime not present.\n");
		std::exit(EXIT_FAILURE);
	}
	return runtime;
}

SnpeInfer::~SnpeInfer()
{
	this->showTime();
	if (snpe_ != nullptr)
	{
		snpe_.reset(nullptr);
	}
}

bool SnpeInfer::initDevice(int device)
{
	std::unique_ptr<zdl::DlContainer::IDlContainer> container;
	container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlc_path));
	zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

	switch (device)
	{
	case InfereDevice::CPU:
		runtime = zdl::DlSystem::Runtime_t::CPU;
		return true;
	case InfereDevice::GPU:
		runtime = zdl::DlSystem::Runtime_t::GPU;
		return true;
	case InfereDevice::DSP:
		runtime = zdl::DlSystem::Runtime_t::DSP;
		return true;
	case InfereDevice::APU:
		runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
		return true;
	}
	return false;
}

bool SnpeInfer::Init(const std::string &model_path,
					 zdl::DlSystem::Runtime_t runtime,
					 std::vector<std::string> output_names)
{
	checkRuntime(runtime);
	// Run in higher clock and provides better performance than POWER_SAVER.
	zdl::DlSystem::PerformanceProfile_t profile = zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE;

	container_ = zdl::DlContainer::IDlContainer::open(model_path);
	if (container_ == nullptr)
	{
		printf("Error while opening the container file.");
		return EXIT_FAILURE;
	}

	zdl::SNPE::SNPEBuilder snpeBuilder(container_.get());
	zdl::DlSystem::PlatformConfig platformConfig;

	// get output names
	zdl::DlSystem::StringList outputTensorNames;
	for (auto str : output_names)
	{
		outputTensorNames.append(str.c_str());
	}

	zdl::DlSystem::RuntimeList runtimeList;
	runtimeList.add(runtime);

	snpe_ = snpeBuilder.setOutputLayers({})
				.setOutputTensors(outputTensorNames)
				.setRuntimeProcessorOrder(runtimeList)
				.setPerformanceProfile(profile)
				.setPlatformConfig(platformConfig)
				.setProfilingLevel(zdl::DlSystem::ProfilingLevel_t::OFF)
				//  .setCPUFallbackMode(false)
				.setUseUserSuppliedBuffers(true)
				.setInitCacheMode(true)
				.build();

	if (nullptr == snpe_.get())
	{
		const char *errStr = zdl::DlSystem::getLastErrorString();
		printf("SNPE build failed: {%s}", errStr);
		return EXIT_FAILURE;
	}
	printf("create snpe  Builder \n");

	// get input tensor names of the network that need to be populated create SNPE
	// user buffers for each application storage buffer for input buffers
	const auto &inputNamesOpt = snpe_->getInputTensorNames();
	if (!inputNamesOpt)
	{
		throw std::runtime_error("Error obtaining input tensor names");
	}

	const zdl::DlSystem::StringList &inputNames = *inputNamesOpt;

	for (const char *name : inputNames)
	{
		printf("Get input name : %s \n", name);
		input_nbbinding_name_.push_back(name);
		// get attributes of buffer by name
		auto bufferAttributesOpt = snpe_->getInputOutputBufferAttributes(name);
		if (!bufferAttributesOpt)
		{
			printf("Error obtaining attributes for input tensor: %s", name);
			return false;
		}

		const zdl::DlSystem::TensorShape &bufferShape = (*bufferAttributesOpt)->getDims();
		printf("Input buffer dims :  %d,  %d , %d ,  %d\n", bufferShape[0], bufferShape[1], bufferShape[2], bufferShape[3]);
		std::vector<size_t> tensorShape;
		for (size_t j = 0; j < bufferShape.rank(); j++)
		{
			tensorShape.push_back(bufferShape[j]);
		}
		m_inputShapes.emplace(name, tensorShape);
	}

	// get output tensor names of the network that need to be populated
	// create SNPE user buffers for each application storage buffer
	// for output buffers
	const auto &outputNamesOpt = snpe_->getOutputTensorNames();

	if (!outputNamesOpt)
		throw std::runtime_error("Error obtaining output tensor names");

	const zdl::DlSystem::StringList &outputNames = *outputNamesOpt;
	for (const char *name : outputNames)
	{
		printf("Get output Names : %s  \n", name);
		m_output_nbbinding_name.push_back(name);
		// get attributes of buffer by name
		auto bufferAttributesOpt = snpe_->getInputOutputBufferAttributes(name);
		if (!bufferAttributesOpt)
		{
			printf("Error obtaining attributes for input tensor: %s", name);
			return false;
		}

		const zdl::DlSystem::TensorShape &bufferShape = (*bufferAttributesOpt)->getDims();
		std::vector<size_t> tensorShape;

		dst_h = bufferShape[0];
		dst_w = bufferShape[1];
		for (size_t j = 0; j < bufferShape.rank(); j++)
		{
			printf("  %d  ", bufferShape[j]);
			tensorShape.push_back(bufferShape[j]);
		}
		printf("\n");
		m_outputShapes.emplace(name, tensorShape);
	}

	createOutputBufferMap(m_outputUserBufferMap, m_applicationOutputBuffers, m_outputUserBuffers, snpe_, false, 8);
	createInputBufferMap(m_inputUserBufferMap, m_applicationInputBuffers, m_inputUserBuffers, snpe_, false, 0);
	InitOutputDimension();
	printf("SNPE Inference init end  \n");
	return true;
}

void SnpeInfer::showTime()
{
	printf("total image number is :   %d \n", count_num);
	printf("infer time:  %.4f  ms, mean time is : %.4f\n", total_time, total_time / count_num);
	printf("postproecess time:  %.4f   ms , mean time is : %.4f\n", process_time, process_time / count_num);
	printf("per image mean time is : %.4f\n", (process_time + total_time) / count_num);
}

void SnpeInfer::saveBin(std::string bin_file)
{
	int index = 0;
	for (auto name : m_output_nbbinding_name)
	{
		auto bufferPtr = m_outputUserBufferMap.getUserBuffer(name.c_str());

		std::cout << "output node : " << name << " total  number : " << m_applicationOutputBuffers.size() << std::endl;
		int buffer_size = (bufferPtr->getSize() / 4);

		std::cout << "output node : " << buffer_size << std::endl;
		float data[buffer_size];
		for (int k = 0; k < bufferPtr->getSize() / 4; k++)
		{
			data[k] = *(outputs_host_[index] + k);
		}

		std::string bin_name = bin_file + name + ".bin";
		// save
		FILE *fp = fopen(bin_name.c_str(), "wb");
		fwrite(data, sizeof(float), sizeof(data), fp);
		fclose(fp);

		bin_name = bin_file + name + ".txt";
		std::ofstream out_txt_file;
		out_txt_file.open(bin_name.c_str(), std::ios::out | std::ios::trunc);

		for (int k = 0; k < bufferPtr->getSize() / 4; k++)
		{
			data[k] = *(outputs_host_[index] + k);
			out_txt_file << data[k] << std::endl;
		}
		out_txt_file.close();
		index++;
	}
}

size_t SnpeInfer::output_size(size_t output_index) const
{
	return output_size_[output_index];
}

std::vector<size_t> SnpeInfer::getOutputShape(const std::string &name)
{
	if (m_outputShapes.find(name) != m_outputShapes.end())
	{
		return m_outputShapes.at(name);
	}
	printf("Can't find any ouput layer named %s", name.c_str());
	return {};
}

void SnpeInfer::InitOutputDimension()
{
	if (!m_output_nbbinding_name.size())
	{
		printf("m_output_nbbinding_name is null!");
		return;
	}

	outputs_host_.resize(m_output_nbbinding_name.size());
	int out_memory_size = 0;

	for (auto output_name : m_output_nbbinding_name)
	{
		auto bufferPtr = m_outputUserBufferMap.getUserBuffer(output_name.c_str());
		if (nullptr == bufferPtr)
		{
			printf("Faild to find output buffer name %s.", output_name.c_str());
		}
		auto output_dims = getOutputShape(output_name);
		output_dimension_.push_back(output_dims);

		output_size_.push_back(bufferPtr->getSize() / sizeof(float));
		out_memory_size += bufferPtr->getSize();
	}

	outputs_host_[0] = new float[out_memory_size];
	for (int i = 1; i < outputs_host_.size(); ++i)
	{
		outputs_host_[i] = (float *)outputs_host_[i - 1] + output_size(i - 1);
	}
}

bool SnpeInfer::PreProcess(const cv::Mat &img, int width, int height)
{
	cv::Mat image_trans;
	// for nv21/nv12 channel is 1
	img.convertTo(image_trans, CV_32FC1);

	// for uyvy/yuyv channel is 2
	// img.convertTo(image_trans, CV_32FC1);

	for (int i = 0; i < input_nbbinding_name_.size(); i++)
	{
		memcpy(&m_applicationInputBuffers.at(input_nbbinding_name_[i].c_str())[0],
			   image_trans.data, img.rows * img.cols * img.channels() * sizeof(float));
	}

	return true;
}

void SnpeInfer::DoInference(const cv::Mat &inputs, size_t batch_size, bool preprocess, std::string filename)
{
	cv::Mat image = inputs; // cv ::Mat(inputs[0].height, inputs[0].width, CV_8UC3, inputs[0].ptr);
	count_num += 1;

	auto process_start = std::chrono::high_resolution_clock::now();
	SnpeInfer::PreProcess(image, input_w, input_h);
	auto process_end = std::chrono::high_resolution_clock::now();
	if (!snpe_->execute(m_inputUserBufferMap, m_outputUserBufferMap))
	{
		printf("SnpeInferer execute failed: %s \n", zdl::DlSystem::getLastErrorString());
		return;
	}

	std::map<std::string, std::vector<float>> output_result;

	int index = 0;
	for (auto name : m_output_nbbinding_name)
	{
		auto bufferPtr = m_outputUserBufferMap.getUserBuffer(name.c_str());
		if (nullptr == bufferPtr)
		{
			printf("Faild to find output buffer name %s.", name.c_str());
		}
		memcpy(outputs_host_[index], &m_applicationOutputBuffers.at(name)[0], bufferPtr->getSize());
		index++;
	}

	auto infer_end = std::chrono::high_resolution_clock::now();
	auto postprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start);
	auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - process_end);
	total_time += infer_time.count();
	process_time += postprocess_time.count();

	// saveBin(filename);

	// // show results
	index = 0;
	for (auto name : m_output_nbbinding_name)
	{
		auto bufferPtr = m_outputUserBufferMap.getUserBuffer(name.c_str());

		// save result to rgb image
		cv::Mat c = cv::Mat(dst_h, dst_w, CV_32FC3, outputs_host_[index]);
		size_t inputHeight = c.rows;
		size_t inputWidth = c.cols;
		size_t channel = c.channels();

		std::string save_file = std::to_string(count_num) + ".png";
		printf("save file :  %s \n", save_file.c_str());
		cv::imwrite(save_file, c);
		index++;
	}
}