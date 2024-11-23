# include <main.hpp>

# include <stdio.h>
# include <stdlib.h>
# include <sys/stat.h>

# include <iostream>
# include <chrono>

# include <torch/torch.h>
# include <torch/script.h>
# include <ATen/cuda/CUDAContext.h>
# include <c10/cuda/CUDACachingAllocator.h>
# include <c10/cuda/CUDAGuard.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>
# include <nvToolsExt.h>
# include <cuda_profiler_api.h>

# include <tests.hpp>
// # include <ctx.hpp>
# include <schd.hpp>
# include <resnet.hpp>

using namespace std;
using namespace chrono;
using namespace torch;
using namespace torch::nn;
using namespace FGPRS;

# define MAX_CONTEXT_COUNT	5
# define MODULE_COUNT				30
# define MODULE_STEP				5

shared_ptr<ResNet<BasicBlock>> mdi[MODULE_COUNT];
Tensor ini[MODULE_COUNT];
MyContext* ctx[MAX_CONTEXT_COUNT];

void dummThread(int index, int maxContext, int timer, double* fps);

void testConcurrency(char** argv)
{
	NoGradGuard no_grad;
	bool stop = false;
	std::thread th[MODULE_COUNT];
	double fps[MODULE_COUNT];
	int mode, contextCount;

	int smCount, timer, index = -1;

	mode = atoi(argv[0]);
	contextCount = atoi(argv[1]);
	timer = atoi(argv[2]);

	switch (mode)
	{
		case 1:
			smCount = 68 / contextCount + 68 % contextCount;
			break;

		case 2:
			smCount = (int)(68 * 1.5) / contextCount + (int)(68 * 1.5) % contextCount;
			break;

		case 3:
			smCount = 68 * 2 / contextCount + 68 * 2 % contextCount;
			break;

		case 4:
			smCount = 68;
			break;
	}

	smCount += smCount % 2;
	smCount = min(smCount, 68);

	int leastPriority, greatestPriority;

	cudaError_t cudaStatus = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaDeviceGetStreamPriorityRange failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		// return 1;
	}

	std::cout << "Least Priority: " << leastPriority << std::endl;
	std::cout << "Greatest Priority: " << greatestPriority << std::endl;

	printf("Running \"Concurrency\" Simulation\n\tMode: %2d\n\tCount: %d\n\tSMs: %3d\n", mode, contextCount, smCount);

	for (int i = 0; i < MODULE_COUNT; i++)
	{
		mdi[i] = resnet18(1000);
		ini[i] = torch::randn({ 1, 3, 224, 224 }, kCUDA);
		mdi[i]->eval();
		mdi[i]->to(kCUDA);
	}

	int options[MAX_CONTEXT_COUNT];

	for (int i = 0; i < MAX_CONTEXT_COUNT; i++)
		options[i] = smCount;

	Tensor dummy;
	steady_clock::time_point tstart, now, tend;
	tstart = steady_clock::now();
	tend = tstart + milliseconds(1000);

	cout << "-------------------------------------------------------------------------------\n";
	cout << "Warming up\n";

	// while (true)
	// {
	// 	for (int j = 0; j < MODULE_COUNT; j++)
	// 	{
	// 		mdi[j]->forward(ini[j]);
	// 		cuCtxSynchronize();
	// 	}

	// 	now = steady_clock::now();

	// 	if (tend <= steady_clock::now())
	// 		break;
	// }

	// cudaProfilerStart();
	// nvtxRangePush("whole");

	if (!Scheduler::initialize(options, max(contextCount, 2), PROPOSED_SCHEDULER, true))
	{
		cout << "CUDA initialization failed.\n";
		return;
	}

	cout << "Count: " << contextCount << "\n";
	cout << "Max Context: " << max(contextCount, 2) << "\n";

	MyContext::selectDefault();

	for (int i = 0; i < contextCount; i++)
	{
		ctx[i] = &Scheduler::_contextPool[i];
		cout << "Context " << i << " has SMs " << ctx[i]->index << endl;
	}

	while (true)
	{
		for (int j = 0; j < MODULE_COUNT; j++)
		{
			for (int i = 0; i < contextCount; i++)
			{
#ifdef ENABLE_NVTX_PROFILING
				auto id = nvtxRangeStartA(string("Context: " + to_string(i)).c_str());
#endif

				ctx[i]->select();

				for (int k = 0; k < 2; k++)
				{
					auto outDummy = mdi[j]->forward(ini[j]);
					cuCtxSynchronize();
				}

				ctx[i]->release();

#ifdef ENABLE_NVTX_PROFILING
				nvtxRangeEnd(id);
#endif
			}
		}

		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}

	steady_clock::time_point t1, t2;
	duration<double> d;
	vector<double> results(MODULE_COUNT / MODULE_STEP);

	cudaProfilerStart();

#ifdef ENABLE_NVTX_PROFILING
	nvtxRangePush("whole");
#endif

	for (int j = MODULE_STEP; j <= MODULE_COUNT; j += MODULE_STEP)
	{
		cout << "Running with " << j << " network(s): ";

		Scheduler::selectDefaultContext();

#ifdef ENABLE_NVTX_PROFILING
		auto id = nvtxRangeStartA(string("networks: " + to_string(j)).c_str());
#endif

		for (int i = 0; i < j; i++)
		{
			th[i] = std::thread(dummThread, i, contextCount, timer, &fps[i]);
			// this_thread::sleep_for(microseconds(1000));
		}

		results[j / MODULE_STEP - 1] = 0;

		for (int i = 0; i < j; i++)
		{
			th[i].join();
			results[j / MODULE_STEP - 1] += fps[i];
		}

		this_thread::sleep_for(microseconds(5000));

#ifdef ENABLE_NVTX_PROFILING
		nvtxRangeEnd(id);
#endif

		printf("%6.3lf fps\n", results[j / MODULE_STEP - 1]);
	}

#ifdef ENABLE_NVTX_PROFILING
	nvtxRangePop();
	cudaProfilerStop();
#endif

	cout << "Saving results\n";
	writeToFile(string("concurrency" + to_string(mode)).c_str(), contextCount, results);
	cout << "-------------------------------------------------------------------------------\n\n";
}

void dummThread(int index, int maxContext, int timer, double* fps)
{
	int count = 0;
	int ctxIndex = ctx[index % maxContext]->index;
	// cout << "ctxIndex: " << ctxIndex << "\n";
	int actualIndex = ctxIndex - 1;
	ctx[actualIndex]->select();
	c10::cuda::CUDAStream::setContextIndex(ctxIndex);
	auto str = c10::cuda::getStreamFromPool(index % 2 == 1, ctxIndex);

	// cout << "Priority: " << str.priority() << endl;
	// cout << "Thread " << index << " is running on context " << ctxIndex << endl;
	// cout << "Context " << ctx[actualIndex]->index << " has SMs " << ctx[actualIndex]->smCount << endl;

	while (str.isBusy())
		str = c10::cuda::getStreamFromPool(index % 2 == 1, ctxIndex);

	str.select();
	at::cuda::setCurrentCUDAStream(str);

	// auto id = nvtxRangeStartA(string("index: " + to_string(index) + "\tCtx: " + to_string(ctxIndex)).c_str());

	steady_clock::time_point now;
	auto tstart = steady_clock::now();
	auto tend = tstart + milliseconds(timer);

	while (true)
	{
#ifdef ENABLE_NVTX_PROFILING
		auto id2 = nvtxRangeStartA(string("repeat: " + to_string(count + 1)).c_str());
#endif

		mdi[index]->forward(ini[index]);
		// cuCtxSynchronize();
		str.synchronize();
		count++;

		now = steady_clock::now();

#ifdef ENABLE_NVTX_PROFILING
		nvtxRangeEnd(id2);
#endif

		if (tend <= now)
			break;
	}

	// nvtxRangeEnd(id);
	str.release();
	duration<double> d = now - tstart;
	*fps = count / d.count();
	ctx[actualIndex]->release();
	// cout << "FPS(" << index << "): " << *fps << "\n";
	// cout << "Priority: " << (str.priority() == 0 ? "low" : "high") << "\tFPS: " << *fps << "\n";
	// cout << "Count: " << count << "\tFPS: " << *fps << "\n";
}