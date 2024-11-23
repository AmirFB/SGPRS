# include <main.hpp>

# include <iostream>
# include <fstream>
# include <iomanip>
# include <thread>
# include <pthread.h>
# include <chrono>
# include <string>
# include <cstdlib>
# include <future>
# include <sys/stat.h>
# include <random>
# include <ctime>
# include <filesystem>
# include "spdlog/spdlog.h"
# include "spdlog/sinks/basic_file_sink.h"

# include <torch/torch.h>
# include <torch/script.h>
# include <c10/cuda/CUDAStream.h>
# include <ATen/cuda/CUDAContext.h>
# include <c10/cuda/CUDACachingAllocator.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>
# include <cuda_profiler_api.h>
# include <nvToolsExt.h>

# include <loop.hpp>
# include <schd.hpp>
# include <cnt.hpp>

# include <cif10.hpp>
# include <resnet.hpp>

# include <tests.hpp>

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <sys/time.h>
# include <sched.h>
# include <deeplab.hpp>

# include <c10/cuda/CUDACachingAllocator.h>

using namespace std;
using namespace chrono;
using namespace spdlog;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

# define PRELIMINNARY 0
# define SCHEDULER 		1
# define MODE					SCHEDULER

# define MAX_MODULE_COUNT		30

shared_ptr<logger> logger;

void distributeSMs(int* array, int total, int count);
vector<double> generateUtilization(int count, double total);

int main(int argc, char** argv)
{
# if MODE == SCHEDULER
	srand(time(nullptr));
	auto logger = spdlog::basic_logger_mt("main_logger", "log.log");
	logger->set_pattern("[%S.%f] %v");
	NoGradGuard no_grad;
	int level = 3;
	int moduleCount, dummyCount = 0;
	double frequency;

	SchedulerType type;
	int* smOptions;
	int smCount;
	int mode;
	int distribute;
	int maxStreams;
	const int MaxSMs = 68;
	int totalSMs;
	int timer;
	string fileNameComp, fileNameMiss;

	if (!strcmp(argv[1], "proposed") && !strcmp(argv[2], "clear"))
	{
		filesystem::remove_all("results/final");
		return 0;
	}

	// moduleCount = atoi(argv[2]);
	frequency = atof(argv[2]);
	timer = atoi(argv[3]);
	moduleCount = 5;
	cout << "Initializing scheduler ..." << endl;

	if (!strcmp(argv[1], "proposed"))
	{
		type = PROPOSED_SCHEDULER;

		mode = atoi(argv[4]);
		distribute = atoi(argv[5]);
		smCount = atoi(argv[6]);

		if (smCount == 2)
			maxStreams = 3;

		else if (smCount == 3)
			maxStreams = 2;

		else
			maxStreams = 2;

		MyContext::maxParallel = maxStreams;
		dummyCount = min(smCount * maxStreams - 1, moduleCount - 1);

		smOptions = new int[smCount];

		if (mode == 1)
			totalSMs = MaxSMs;

		else if (mode == 2)
			totalSMs = MaxSMs * 1.5;

		else
			totalSMs = MaxSMs * 2;;

		if (distribute == 1)
		{
			for (int i = 0; i < smCount; i++)
			{
				smOptions[i] = ceil((double)totalSMs / smCount);
				smOptions[i] = min(smOptions[i] + smOptions[i] % 2, MaxSMs);
			}
		}

		else
		{
			int divider = smCount * (smCount + 1) / 2;

			for (int i = 0; i < smCount; i++)
			{
				smOptions[i] = ceil((i + 1) * totalSMs / divider);
				smOptions[i] = min(smOptions[i] + smOptions[i] % 2, MaxSMs);
			}
		}

		totalSMs = 0;

		for (int i = 0; i < smCount; i++)
			totalSMs += smOptions[i];

		cout << "Simulation Parameters:" << endl
			<< "\tMode: " << (mode == 1 ? "1.0x" : (mode == 2 ? "1.5x" : "2.0x")) << endl
			<< "\tDistribute: " << (distribute == 1 ? "Equal" : "Stepwise") << endl
			<< "\tSM Count: " << smCount << endl
			<< "\tMax Streams: " << maxStreams << endl
			<< "\tTotal SMs: " << totalSMs << endl
			<< "----------------------------------------" << endl;

		filesystem::create_directory("results/final");
		filesystem::create_directory(("results/final/" + to_string(smCount)).c_str());
		fileNameComp = ("final/" + to_string(smCount) + "/" + (mode == 1 ? "1.0x" : (mode == 2 ? "1.5x" : "2.0x")) + "_" + (distribute == 1 ? "Equal" : "Stepwise") + "_Comp.csv");
		fileNameMiss = ("final/" + to_string(smCount) + "/" + (mode == 1 ? "1.0x" : (mode == 2 ? "1.5x" : "2.0x")) + "_" + (distribute == 1 ? "Equal" : "Stepwise") + "_Miss.csv");

		writeToFile(fileNameComp, mode, true, true);
		writeToFile(fileNameComp, distribute, true, false);

		writeToFile(fileNameMiss, mode, true, true);
		writeToFile(fileNameMiss, distribute, true, false);
	}

	else if (!strcmp(argv[1], "mps"))
	{
		type = MPS_SCHEDULER;
		smCount = atoi(argv[4]);
		smOptions = new int[smCount];

		for (int i = 0; i < smCount; i++)
			smOptions[i] = 68;

		filesystem::create_directory("results/final");
		filesystem::create_directory(("results/final/" + to_string(smCount)).c_str());
		fileNameComp = ("final/" + to_string(smCount) + "/Naive_Comp.csv");
		fileNameMiss = ("final/" + to_string(smCount) + "/Naive_Miss.csv");

		writeToFile(fileNameComp, 0, true, true);
		writeToFile(fileNameComp, 0, true, false);

		writeToFile(fileNameMiss, 0, true, true);
		writeToFile(fileNameMiss, 0, true, false);
	}

	else if (!strcmp(argv[1], "pmps"))
	{
		type = PMPS_SCHEDULER;
		smCount = moduleCount;

		smOptions = new int[smCount];
		distributeSMs(smOptions, 68, smCount);
	}

	else if (!strcmp(argv[1], "pmpso"))
	{
		type = PMPSO_SCHEDULER;
		smCount = moduleCount;
		smOptions = new int[smCount];
		distributeSMs(smOptions, max(68 * (smCount / 2 + 0.5), 68.0), smCount);
	}

	else if (!strcmp(argv[1], "nomps"))
	{
		type = NOMPS_SCHEDULER;
		smCount = 1;
		smOptions = new int[1] {68};
	}

	Scheduler::initialize(smOptions, smCount, type, true);
	MyContext::selectDefault();

	Tensor inputs[MAX_MODULE_COUNT];
	shared_ptr<MyContainer> mods[MAX_MODULE_COUNT];
	Loop loops[MAX_MODULE_COUNT];

	cout << "Initializing modules ..." << endl;
	filesystem::remove_all("logs");

	string name, freqStr;

	// cudaProfilerStart();

	for (int i = 0; i < MAX_MODULE_COUNT; i++)
	{
		stringstream stream;
		stream << fixed << setprecision(2) << frequency;
		freqStr = stream.str();

		name = "resnet" + to_string(i + 1);

		inputs[i] = torch::randn({ 1, 3, 224, 224 }, kCUDA);
		mods[i] = resnet18(1000);

		loops[i] = Loop(name, mods[i], frequency, i);
		loops[i].prepare();
	}

	for (size_t i = 0; i < MAX_MODULE_COUNT; i++)
		Scheduler::dummyContainer.push_back(DummyContainer{ mods[i], &inputs[i], i });

	random_shuffle(Scheduler::dummyContainer.begin(), Scheduler::dummyContainer.end());
	Scheduler::dummyContainer.resize(dummyCount + 1);

	// cudaProfilerStart();

	for (int i = 0; i < MAX_MODULE_COUNT; i++)
	{
		cout << "Initializing " << mods[i]->_name << " ..." << endl;
		loops[i].initialize(smCount - 1, inputs[i], type, level);
	}

	// cudaProfilerStop();
	// return 0;

	cout << "Initializing modules finished!" << endl << endl << endl << endl << endl;

	cout << "Warming up ..." << endl;

	cout << "Memory: " << Scheduler::getFreeMemoryGB() << " GB" << endl;
	// cudaProfilerStart();

	for (int i = 0; i < MAX_MODULE_COUNT; i++)
		loops[i].start(&inputs[i], type, level, false, 100);

	// this_thread::sleep_for(milliseconds(100));

	for (int i = 0; i < MAX_MODULE_COUNT; i++)
		loops[i].stop();

	for (int i = 0; i < MAX_MODULE_COUNT; i++)
		loops[i].wait();

	cout << endl << endl << endl << endl << endl;

	cout << "Memory: " << Scheduler::getFreeMemoryGB() << " GB" << endl;
	cout << "Here we go ...\n";

	logger->info("Started!");

	cudaProfilerStart();
	nvtxRangePush("whole");

	double total = 0, comp = 0, miss = 0;
	double compPercent, missPercent;

	for (int j = 15; j <= MAX_MODULE_COUNT; j++)
	{
		cout << "........................\n";
		cout << "Running with " << j << " modules:\n";

		for (int i = 0; i < j; i++)
			loops[i].start(&inputs[i], type, level, false, timer);

		// this_thread::sleep_for(milliseconds((int)(1000 + 1000 / freq)));

		// for (int i = 0; i < j; i++)
		// 	loops[i].stop();

		total = 0;
		comp = 0;
		miss = 0;

		for (int i = 0; i < j; i++)
		{
			loops[i].wait();

			total += loops[i].totalCount;
			comp += loops[i].compCount;
			miss += loops[i].missCount;
		}

		compPercent = 100. * comp / total;
		missPercent = 100. * miss / total;

		writeToFile(fileNameComp, compPercent, j != MAX_MODULE_COUNT, false);
		writeToFile(fileNameMiss, missPercent, j != MAX_MODULE_COUNT, false);
		cout << "fileNames: " << fileNameComp << " " << fileNameMiss << endl;
	}

	cout << "........................\n";

	nvtxRangePop();
	cudaProfilerStop();

	logger->info("Finished!");
	this_thread::sleep_for(milliseconds(100));
	torch::cuda::synchronize();
	cudaDeviceSynchronize();
	cout << "-------------------------------------------------------------------------------\n\n";

# elif MODE == PRELIMINARY

	char* op = argv[1];
	mkdir("results", 0777);

	if (!strcmp(op, "clear"))
	{
		cout << "Removing previous results of \"" << argv[2] << "\" simulation\n";

		if (!strcmp(op, "concurrency"))
		{
			remove(string("results/concurrency1.csv").c_str());
			remove(string("results/concurrency2.csv").c_str());
			remove(string("results/concurrency3.csv").c_str());
			remove(string("results/concurrency4.csv").c_str());
			remove(string("results/concurrency5.csv").c_str());
		}

		else
			remove((string("results/") + string(argv[2]) + ".csv").c_str());
	}

	else if (!strcmp(op, "speedup"))
		testSpeedup(&argv[2]);

	else if (!strcmp(op, "concurrency"))
		testConcurrency(&argv[2]);

# endif
}

void distributeSMs(int* array, int total, int count)
{
	int minPer, rem;

	minPer = total / count - (total / count) % 2;
	rem = total - count * minPer;

	for (int i = 0; i < count; i++)
	{
		array[i] = minPer;

		if (rem > 0)
		{
			array[i] += 2;
			rem -= 2;
		}
	}
}

vector<double> generateUtilization(int count, double total)
{
	// vector<double> result(count);
	// random_device rd;
	// std::mt19937 gen(rd());
	// uniform_real_distribution<double> dis(0.05, 0.9);

	// double sum = 0;

	// for (int i = 0; i < count; i++)
	// {
	// 	result[i] = dis(gen);
	// 	sum += result[i];
	// }

	// for (int i = 0; i < count; i++)
	// 	result[i] = result[i] / sum * total;

	// return result;
}