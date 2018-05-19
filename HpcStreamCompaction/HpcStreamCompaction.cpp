// HpcStreamCompaction.cpp: Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <functional>

#pragma region prefix_methods
std::vector<int> generate_vector(int min, int max, int size);
std::vector<int> prefixsum(std::vector<int> problem);
std::vector<int> prefixsum_sequential(std::vector<int> input);
std::vector<int> prefixsum_gpu(std::vector<int> input);
std::vector<int> prefixsum_workefficient(std::vector<int> input, std::vector<int> &blockSums, int BLOCK_SIZE);
void addBlockSums(std::vector<int> &output, std::vector<int> blockSums);
std::vector<int> opencl_filter(std::vector<int> input, const std::string filter, const int comparable);
std::vector<int> filter_gpu(std::vector<int> input, const int filter, const int argument);
std::vector<int> filter_sequential(std::vector<int> input, const int filter, const int argument);
std::vector<int> opencl_scatter(std::vector<int> input, std::vector<int> addresses);
#pragma endregion

#pragma region global
cl::Platform platform;
cl::Program program;
cl::Context context;
cl_int err = CL_SUCCESS;
std::vector<cl::Device> devices;
#pragma endregion

#pragma region helpers
void print_vector(std::vector<int> vec, std::string info);
void print_OpenCl_Error(cl::Error err);
bool setup_OpenCl_Platform();
void print_OpenCl_Platform(cl::Platform platform);
#pragma endregion

#pragma region CONSTANTS
const bool DEBUG = false;
const bool PRINT = false;

const int BLOCK_SIZE = 32;
const int MAX_WORK_GROUP_SIZE = 1024;
const int RANGE = 10;
const std::string KERNEL_FILE = "kernel.cl";
const std::vector<std::string> KERNEL_FILTERS = { "", "filterLess", "filterEqual", "filterGreater" };
#pragma endregion

#pragma region region options
int PROBLEM_SIZE = BLOCK_SIZE * MAX_WORK_GROUP_SIZE;
int FILTER;
int ARGUMENT;
#pragma endregion
int main(int argc, char* argv[])
{
	if (argc == 4) {
		int FACTOR = strtol(argv[1], nullptr, 0);
		PROBLEM_SIZE *= FACTOR;
		FILTER = strtol(argv[2], nullptr, 0);
		ARGUMENT = strtol(argv[3], nullptr, 0);
	}
	else {
		int FACTOR;
		std::cout << "Enter factor for problem size (" << BLOCK_SIZE << " x " << MAX_WORK_GROUP_SIZE << ")" << std::endl;
		std::cin >> FACTOR;
		PROBLEM_SIZE *= FACTOR;
		
		std::cout << "Select the filter type (1 = greater than, 2 = less than, 3 = equal)" << std::endl;
		std::cin >> FILTER;
		
		std::cout << "Enter the filter argument: " << std::endl;
		std::cin >> ARGUMENT;
	}


	std::cout << "Arguments provided: " << std::endl;
	std::cout << "FILTER: " << FILTER << ", ARGUMENT: " << ARGUMENT << " ,PROBLEM_SIZE: " << PROBLEM_SIZE << std::endl;

	if (setup_OpenCl_Platform()) {
		auto vec = generate_vector(1, 10, PROBLEM_SIZE);
		if(PRINT)
			print_vector(vec, "Original");
		auto scattered_seq = filter_sequential(vec, FILTER, ARGUMENT);
		auto scattered_gpu = filter_gpu(vec, FILTER, ARGUMENT);
		std::cout << "Sequentially filtered result size: " << scattered_seq.size() << std::endl;
	}
    return EXIT_SUCCESS;
}

std::vector<int> filter_gpu(std::vector<int> input, const int filter, const int argument) {
	auto TIME_START = std::chrono::high_resolution_clock::now();
		
	/* Phase 1 - create temporary vector with conditional filter flags */
	auto temp = opencl_filter(input, KERNEL_FILTERS[filter], argument);
	if(DEBUG)
		print_vector(temp, "temp");
	/* Phase 2 - prefixsum of temporary filter vector */
	auto addresses = prefixsum(temp);
	if(DEBUG)
		print_vector(addresses, "scanned addresses");
	/* Phase 3 - scatter input elements to output vector */
	auto scattered = opencl_scatter(input, addresses);
	if(DEBUG)
		print_vector(scattered, "scattered");
	
	auto TIME_END = std::chrono::high_resolution_clock::now();
	auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
	std::cout << "GPU Filter Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
	std::cout << "**********************************************" << std::endl;
	return scattered;
}

std::vector<int> filter_sequential(std::vector<int> input, const int filter, const int argument) {
	
	auto TIME_START = std::chrono::high_resolution_clock::now();
	std::vector<int> filtered;
	std::function<bool(int)> pred;
	switch (filter) {
		case 1: pred = [](int i) { return i < ARGUMENT; }; break;
		case 2: pred = [](int i) { return i > ARGUMENT; }; break;
		case 3: pred = [](int i) { return i == ARGUMENT; }; break;
		default: pred = [](int i) { return i < ARGUMENT; };
	}
	for (auto e: input) {
		if (pred(e))
			filtered.push_back(e);
	}
	if(DEBUG)
		print_vector(filtered, "Sequential filter");
	
	auto TIME_END = std::chrono::high_resolution_clock::now();
	auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
	std::cout << "Sequential Filter Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
	std::cout << "**********************************************" << std::endl;
	return filtered;
}

std::vector<int> opencl_filter(std::vector<int> input, const std::string filter, const int comparable) {
	
	const size_t _size = input.size();
	std::vector<int> filtered(_size);
	try {
		cl::CommandQueue queue(context, devices[0], 0, &err);
		auto TIME_START = std::chrono::high_resolution_clock::now();

		cl::Buffer input_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, _size * sizeof(cl_int));
		cl::Buffer output_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, _size * sizeof(cl_int));

		queue.enqueueWriteBuffer(
			input_buffer,
			CL_TRUE,
			0,
			_size * sizeof(cl_int),
			&input[0]
		);

		const std::string KERNEL = filter;
		cl::Kernel addKernel(program, KERNEL.c_str(), &err);


		addKernel.setArg(0, input_buffer);
		addKernel.setArg(1, output_buffer);
		addKernel.setArg(2, comparable);

		cl::NDRange global(_size);

 		if (DEBUG)
			std::cout << "CALL KERNEL " << KERNEL << std::endl;

		queue.enqueueNDRangeKernel(addKernel, 0, global);

		queue.finish();

		cl::Event event;
		queue.enqueueReadBuffer(
			output_buffer,
			CL_TRUE,
			0,
			_size * sizeof(cl_int),
			&filtered[0],
			NULL,
			&event
		);

		event.wait();
		auto TIME_END = std::chrono::high_resolution_clock::now();

		if (PRINT) {
			std::cout << "Received data from kernel" << std::endl;
			auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
			std::cout << "GPU Filter Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
			std::cout << "**********************************************" << std::endl;
		}
		if (DEBUG) {
			print_vector(filtered, "filtered vector");
		}
	}
	catch (cl::Error err) {
		print_OpenCl_Error(err);
	}
	return filtered;
}

std::vector<int> opencl_scatter(std::vector<int> input, std::vector<int> addresses) {

	std::cout << "Addresses last: " << addresses[addresses.size() - 1] << std::endl;;
	const size_t _size = input.size();
	std::vector<int> filtered(_size);
	try {
		cl::CommandQueue queue(context, devices[0], 0, &err);
		auto TIME_START = std::chrono::high_resolution_clock::now();

		cl::Buffer input_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, _size * sizeof(cl_int));
		cl::Buffer address_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, _size * sizeof(cl_int));
		cl::Buffer output_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, _size * sizeof(cl_int));

		queue.enqueueWriteBuffer(
			input_buffer,
			CL_TRUE,
			0,
			_size * sizeof(cl_int),
			&input[0]
		);

		queue.enqueueWriteBuffer(
			address_buffer,
			CL_TRUE,
			0,
			_size * sizeof(cl_int),
			&addresses[0]
		);

		const std::string KERNEL = "scatter";
		cl::Kernel addKernel(program, KERNEL.c_str(), &err);


		addKernel.setArg(0, input_buffer);
		addKernel.setArg(1, address_buffer);
		addKernel.setArg(2, output_buffer);

		cl::NDRange global(_size);

		if (DEBUG)
			std::cout << "CALL KERNEL " << KERNEL << std::endl;

		queue.enqueueNDRangeKernel(addKernel, 0, global);

		queue.finish();

		cl::Event event;
		queue.enqueueReadBuffer(
			output_buffer,
			CL_TRUE,
			0,
			_size * sizeof(cl_int),
			&filtered[0],
			NULL,
			&event
		);

		event.wait();
		auto TIME_END = std::chrono::high_resolution_clock::now();

		if (PRINT) {
			std::cout << "Received data from kernel" << std::endl;
			auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
			std::cout << "GPU Filter Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
			std::cout << "**********************************************" << std::endl;
		}
		if (DEBUG) {
			print_vector(filtered, "filtered vector");
		}
	}
	catch (cl::Error err) {
		print_OpenCl_Error(err);
	}
	return filtered;
}

std::vector<int> prefixsum(std::vector<int> problem) {
	return (problem.size() > MAX_WORK_GROUP_SIZE) ? prefixsum_gpu(problem) : prefixsum_sequential(problem);
}

std::vector<int> prefixsum_gpu(std::vector<int> input) {

	if (DEBUG)
		std::cout << "Scanning large vector[" << input.size() << "]" << std::endl;
	std::vector<int> blockSums(input.size() / BLOCK_SIZE);

	auto TIME_START = std::chrono::high_resolution_clock::now();
	/* Phase 1: Scan single Blocks of Problem */
	auto output = prefixsum_workefficient(input, blockSums, BLOCK_SIZE);
	/* Phase 2: Scan Blocksums vector */
	auto blockSumsScanned = prefixsum(blockSums);
	/* Phase 3: Add BlocksumsScanned to Output */
	addBlockSums(output, blockSumsScanned);
	/* Finished ! */
	auto TIME_END = std::chrono::high_resolution_clock::now();
	if (PRINT) {
		auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
		std::cout << "Scanning large array elapsed in " << TOTAL_ELAPSED << " milliseconds" << std::endl;
		std::cout << "**********************************************" << std::endl;
	}
	return output;
}

std::vector<int> prefixsum_workefficient(std::vector<int> input, std::vector<int> &blockSums, int BLOCK_SIZE)
{
	const int _size = input.size();
	std::vector<int> output(_size);

	try {
		cl::CommandQueue queue(context, devices[0], 0, &err);
		auto TIME_START = std::chrono::high_resolution_clock::now();

		cl::Buffer input_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, _size * sizeof(cl_int));
		cl::Buffer output_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, _size * sizeof(cl_int));
		cl::Buffer blocksums_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, blockSums.size() * sizeof(cl_int));

		queue.enqueueWriteBuffer(
			input_buffer,
			CL_TRUE,
			0,
			_size * sizeof(cl_int),
			&input[0]
		);

		const std::string KERNEL = "prefixsum";
		cl::Kernel addKernel(program, KERNEL.c_str(), &err);

		addKernel.setArg(0, input_buffer);
		addKernel.setArg(1, output_buffer);
		int LOCAL_SIZE = sizeof(cl_int) * BLOCK_SIZE;
		addKernel.setArg(2, cl::LocalSpaceArg(cl::Local(LOCAL_SIZE)));
		addKernel.setArg(3, blocksums_buffer);
		addKernel.setArg(4, BLOCK_SIZE);

		const int GLOBAL_WORK_SIZE = _size;
		const int LOCAL_WORK_SIZE = BLOCK_SIZE;
		cl::NDRange global(GLOBAL_WORK_SIZE);
		cl::NDRange local(LOCAL_WORK_SIZE);

		if (DEBUG)
			std::cout << "CALL KERNEL " << KERNEL << std::endl;

		queue.enqueueNDRangeKernel(addKernel, 1, global, local);

		queue.finish();

		cl::Event event;
		queue.enqueueReadBuffer(
			output_buffer,
			CL_TRUE,
			0,
			output.size() * sizeof(cl_int),
			&output[0],
			NULL,
			&event
		);

		queue.enqueueReadBuffer(
			blocksums_buffer,
			CL_TRUE,
			0,
			blockSums.size() * sizeof(cl_int),
			&blockSums[0],
			NULL,
			&event
		);

		event.wait();
		auto TIME_END = std::chrono::high_resolution_clock::now();

		if (PRINT) {
			std::cout << "Received data from kernel" << std::endl;
			auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
			std::cout << "GPU Scan Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
			std::cout << "**********************************************" << std::endl;
		}
		if (DEBUG) {
			print_vector(output, "Output prefixsum ");
			print_vector(blockSums, "blockSums");
		}
	}
	catch (cl::Error err) {
		print_OpenCl_Error(err);
	}
	return output;
}

std::vector<int> prefixsum_sequential(std::vector<int> input)
{
	auto TIME_START = std::chrono::high_resolution_clock::now();
	std::vector<int> scanned = { 0 };
	for (unsigned int i = 1; i < input.size(); i++) {
		scanned.push_back(scanned[i - 1] + input.at(i - 1));
	}
	auto TIME_END = std::chrono::high_resolution_clock::now();
	auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
	if (DEBUG)
		std::cout << "Sequential Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
	if (PRINT)
		print_vector(scanned, "Sequential Scan Result");
	return scanned;
}

void addBlockSums(std::vector<int> &output, std::vector<int> blockSumsScanned)
{
	const size_t _size = output.size();
	if (DEBUG)
		std::cout << "Adding blocksums to prescanned vector ..." << std::endl;
	cl::CommandQueue queue(context, devices[0], 0, &err);
	auto TIME_START = std::chrono::high_resolution_clock::now();

	cl::Buffer output_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, _size * sizeof(cl_int));
	cl::Buffer blockSumsScanned_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, blockSumsScanned.size() * sizeof(cl_int));

	queue.enqueueWriteBuffer(
		output_buffer,
		CL_TRUE,
		0,
		_size * sizeof(cl_int),
		&output[0]
	);

	queue.enqueueWriteBuffer(
		blockSumsScanned_buffer,
		CL_TRUE,
		0,
		blockSumsScanned.size() * sizeof(cl_int),
		&blockSumsScanned[0]
	);

	const std::string KERNEL = "addBlockSums";
	cl::Kernel addKernel(program, KERNEL.c_str(), &err);

	addKernel.setArg(0, output_buffer);
	addKernel.setArg(1, blockSumsScanned_buffer);

	cl::NDRange global(_size);
	cl::NDRange local(BLOCK_SIZE);

	if (DEBUG)
		std::cout << "CALL KERNEL " << KERNEL << std::endl;

	queue.enqueueNDRangeKernel(addKernel, 0, global, local);

	cl::Event event;
	queue.enqueueReadBuffer(
		output_buffer,
		CL_TRUE,
		0,
		_size * sizeof(cl_int),
		&output[0],
		NULL,
		&event
	);

	queue.enqueueReadBuffer(
		blockSumsScanned_buffer,
		CL_TRUE,
		0,
		blockSumsScanned.size() * sizeof(cl_int),
		&blockSumsScanned[0],
		NULL,
		&event
	);
	event.wait();
	auto TIME_END = std::chrono::high_resolution_clock::now();
	if (PRINT) {
		std::cout << "Received data from kernel" << std::endl;
		auto TOTAL_ELAPSED = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END - TIME_START).count();
		std::cout << "GPU Scan Execution: " << TOTAL_ELAPSED << " milliseconds" << std::endl;
		std::cout << "**********************************************" << std::endl;
	}
	if (DEBUG)
		print_vector(output, "Final Scan Result of arbitrary array");

}

void print_vector(std::vector<int> vec, std::string info)
{
	std::cout << info << std::endl;
	int j = 0;
	for (unsigned int i = 0; i < vec.size(); i++)
	{
		std::cout << vec.at(i);
		if (i < vec.size() - 1) {
			std::cout << ", ";
		}
		if (j > 15) {
			std::cout << std::endl;
			j = 0;
		}
		j++;
	}
	std::cout << std::endl;
}

std::vector<int> generate_vector(int min, int max, int size)
{
	std::cout << "**********************************************" << std::endl;
	std::cout << "Generating vector with problem size " << size << std::endl;
	std::vector<int> vec;
	for (int i = 0; i < size; i++)
	{
		vec.push_back(min + (rand() % static_cast<int>(max - min + 1)));
	}
	if (PRINT) {
		std::cout << "**********************************************" << std::endl;
		print_vector(vec, "Original Problem Vector");
	}
	return vec;
}

bool setup_OpenCl_Platform()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cout << "No OpenCL platforms available!\n";
		return false;
	}
	platform = platforms.size() == 2 ? platforms[1] : platforms[0];
	if (DEBUG)
		print_OpenCl_Platform(platform);
	cl_context_properties properties[] =
	{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

	devices = context.getInfo<CL_CONTEXT_DEVICES>();

	std::ifstream sourceFile(KERNEL_FILE);
	if (!sourceFile)
	{
		std::cout << "kernel source file " << KERNEL_FILE << " not found!" << std::endl;
		return false;
	}
	std::string sourceCode(
		std::istreambuf_iterator<char>(sourceFile),
		(std::istreambuf_iterator<char>()));
	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
	program = cl::Program(context, source);
	program.build(devices);
	return true;
}

void print_OpenCl_Platform(cl::Platform platform)
{
	const cl_platform_info attributeTypes[5] = {
		CL_PLATFORM_NAME,
		CL_PLATFORM_VENDOR,
		CL_PLATFORM_VERSION,
		CL_PLATFORM_PROFILE,
		CL_PLATFORM_EXTENSIONS };

	std::cout << "**********************************************" << std::endl;
	std::cout << "Selected Platform Information: " << std::endl;
	std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << "Platform Version: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
	std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
	std::cout << "**********************************************" << std::endl;
}

void print_OpenCl_Error(cl::Error err)
{
	std::string s;
	program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &s);
	std::cout << s << std::endl;
	program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_OPTIONS, &s);
	std::cout << s << std::endl;

	std::cerr
		<< "ERROR: "
		<< err.what()
		<< "("
		<< err.err()
		<< ")"
		<< std::endl;
}
