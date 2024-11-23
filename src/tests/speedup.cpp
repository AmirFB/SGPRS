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

# define LAYER_COUNT	0
# define MODULE_COUNT	5
# define TOTAL_COUNT	6

Sequential mod[MODULE_COUNT];
Tensor in[MODULE_COUNT];
auto net = resnet18(1000);
Tensor inn = torch::randn({ 1, 3, 224, 224 }, kCUDA);
// Tensor inl[MODULE_COUNT - LAYER_COUNT - 1];// = torch::randn({ 1, 64, 56, 56 }, kCUDA);

void forward(int index)
{
	if (index < MODULE_COUNT)
		mod[index]->forward(in[index]);

	// else if (index < (MODULE_COUNT - 1))
	// 	net->forwardL(inl[index - LAYER_COUNT], index - LAYER_COUNT);

	else
		net->forward(inn);
}

void testSpeedup(char** argv)
{
	string moduleName[] = {
		"CV", "FC", "BN", "RL", "MP",
		// "L0", "L1", "L2", "L3", "L4", "LX",
		"RN" };
	NoGradGuard no_grad;

	int smCount, timer, index = -1;

	smCount = atoi(argv[0]);
	timer = (int)(atof(argv[1]));

	printf("Running \"Speedup\" simulation. (SM count: %d)\n", smCount);

	mod[++index] = Sequential(Conv2d(Conv2dOptions(64, 128, 3).stride(1).padding(2)));
	in[index] = torch::randn({ 64, 56, 56 }, kCUDA);

	mod[++index] = Sequential(Linear(512 * 4, 1000));
	in[index] = torch::randn(512 * 4, kCUDA);

	mod[++index] = Sequential(BatchNorm2d(128));
	in[index] = torch::randn({ 1, 128, 56, 56 }, kCUDA);

	mod[++index] = Sequential(ReLU());
	in[index] = torch::randn({ 128, 56, 56 }, kCUDA);

	mod[++index] = Sequential(MaxPool2d(MaxPool2dOptions(2).stride(2).padding(0)));
	in[index] = torch::randn({ 128, 56, 56 }, kCUDA);

	for (int i = 0; i < MODULE_COUNT; i++)
	{
		mod[i]->eval();
		mod[i]->to(kCUDA);
	}

	net->eval();
	net->to(kCUDA);

	int options[] = { smCount };

	if (!Scheduler::initialize(options, 1))
	{
		cout << "CUDA initialization failed.\n";
		return;
	}

	cout << "-------------------------------------------------------------------------------\n";
	cout << "Warming up\n";

	auto ctx = Scheduler::selectContext(smCount);

	Tensor dummy;
	steady_clock::time_point tstart, now, tend;
	tstart = steady_clock::now();
	tend = tstart + milliseconds(timer);

	auto ctxD = Scheduler::selectContext(68);
	ctxD->select();

	while (true)
	{
		for (int j = 0; j < TOTAL_COUNT; j++)
			forward(j);

		cuCtxSynchronize();
		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}

	ctxD->release();
	ctx->select();

	while (true)
	{
		for (int j = 0; j < TOTAL_COUNT; j++)
		{
			forward(j);
			cuCtxSynchronize();
		}

		now = steady_clock::now();

		if (tend <= steady_clock::now())
			break;
	}

	ctx->release();

	steady_clock::time_point t1, t2;
	duration<double> d;
	vector<double> results(TOTAL_COUNT);

	ctx->select();

	cudaProfilerStart();

#ifdef ENABLE_NVTX_PROFILING
	nvtxRangePush("whole");
#endif

	for (int j = TOTAL_COUNT - 1; j >= 0; j--)
	{
		cout << "Running operation \"" << moduleName[j] << "\": ";
		int count = 0;

#ifdef ENABLE_NVTX_PROFILING
		auto id = nvtxRangeStartA(moduleName[j].c_str());
#endif

		tstart = steady_clock::now();
		tend = tstart + milliseconds(timer);

		while (true)
		{
			forward(j);
			cuCtxSynchronize();
			count++;
			now = steady_clock::now();

			if (tend <= steady_clock::now())
				break;
		}

#ifdef ENABLE_NVTX_PROFILING
		nvtxRangeEnd(id);
#endif

		d = now - tstart;
		results[j] = d.count() / count * 1000000;
		printf("%6.3lfus\n", results[j]);

		// CUexecAffinityParam_v1 affinity;
		// cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
		// cout << "Aff: " << affinity.param.smCount.val << endl;
	}


#ifdef ENABLE_NVTX_PROFILING
	nvtxRangePop();
#endif

	cudaProfilerStop();

	ctx->release();
	cout << "Saving results\n";
	writeToFile("speedup", smCount, results);
	cout << "-------------------------------------------------------------------------------\n\n";
}