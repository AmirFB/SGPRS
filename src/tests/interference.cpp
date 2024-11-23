// # include <stdio.h>
// # include <stdlib.h>
// # include <sys/stat.h>

// # include <iostream>
// # include <chrono>

// # include <torch/torch.h>
// # include <torch/script.h>
// # include <ATen/cuda/CUDAContext.h>
// # include <c10/cuda/CUDACachingAllocator.h>
// # include <c10/cuda/CUDAGuard.h>

// # include <cuda.h>
// # include <cudaTypedefs.h>
// # include <cuda_runtime.h>

// # include <tests.hpp>
// // # include <ctx.hpp>
// # include <schd.hpp>
// # include <resnet.hpp>

// using namespace std;
// using namespace chrono;
// using namespace torch;
// using namespace torch::nn;
// using namespace FGPRS;

// # define CONTEXT_COUNT	4
// # define MODULE_COUNT		10

// shared_ptr<ResNet<BasicBlock>> mdi[MODULE_COUNT];
// Tensor ini[MODULE_COUNT];
// MyContext* ctx[CONTEXT_COUNT];

// void dummThread(int index, MyContext* ctx, bool* stop, double* fps);

void testInterference(char** argv)
{
	// 	NoGradGuard no_grad;
	// 	bool stop = false;
	// 	std::thread th[MODULE_COUNT];
	// 	double fps[MODULE_COUNT];
	// 	int mode;

	// 	int smCount, timer, index = -1;

	// 	smCount = atoi(argv[0]);
	// 	timer = (int)(atof(argv[1]));

	// 	printf("Running \"Interference\" simulation. (SM count: %d)\n", smCount);

	// 	for (int i = 0; i < MODULE_COUNT; i++)
	// 	{
	// 		mdi[i] = resnet18(1000);
	// 		ini[i] = torch::randn({ 1, 3, 224, 224 }, kCUDA);
	// 		mdi[i]->eval();
	// 		mdi[i]->to(kCUDA);
	// 	}

	// 	// int options[] = { smCount, 68, 68, 68, 68, 68, 68, 68, 68, 68 };
	// 	int options[CONTEXT_COUNT];

	// 	for (int i = 0; i < CONTEXT_COUNT; i++)
	// 		options[i] = smCount;

	// 	Tensor dummy;
	// 	steady_clock::time_point tstart, now, tend;
	// 	tstart = steady_clock::now();
	// 	tend = tstart + milliseconds(timer);

	// 	cout << "-------------------------------------------------------------------------------\n";
	// 	cout << "Warming up\n";

	// 	while (true)
	// 	{
	// 		for (int j = 0; j < MODULE_COUNT; j++)
	// 		{
	// 			mdi[j]->forward(ini[j]);
	// 			cuCtxSynchronize();
	// 		}

	// 		now = steady_clock::now();

	// 		if (tend <= steady_clock::now())
	// 			break;
	// 	}

	// 	if (!Scheduler::initialize(options, MODULE_COUNT))
	// 	{
	// 		cout << "CUDA initialization failed.\n";
	// 		return;
	// 	}

	// 	for (int i = 0; i < MODULE_COUNT; i++)
	// 		ctx[i] = &Scheduler::_contextPool[i];

	// 	steady_clock::time_point t1, t2;
	// 	duration<double> d;
	// 	vector<double> results(MODULE_COUNT);

	// 	// ctx[0]->select();

	// 	for (int j = 0; j < MODULE_COUNT; j++)
	// 	{
	// 		cout << "Running with " << j << " dummies: ";
	// 		int count = 0;

	// 		stop = false;

	// 		tstart = steady_clock::now();
	// 		tend = tstart + milliseconds(500);

	// 		while (true)
	// 		{
	// 			for (int i = 0; i < (j + 1); i++)
	// 			{
	// 				mdi[i]->forward(ini[i]);
	// 				cuCtxSynchronize();
	// 			}

	// 			now = steady_clock::now();

	// 			if (tend <= steady_clock::now())
	// 				break;
	// 		}

	// 		// if (true)//smCount != 68)
	// 		// {
	// 		for (int i = 0; i < (j + 1); i++)
	// 			th[i] = std::thread(dummThread, i, ctx[i], &stop, &fps[i]);

	// 		// if (j > 0)
	// 		// 	this_thread::sleep_for(milliseconds(100));
	// 	// }

	// 		tstart = steady_clock::now();
	// 		tend = tstart + milliseconds(timer);
	// 		// count = 0;

	// 		// while (true)
	// 		// {
	// 		// 	mdi[j]->forward(ini[j]);
	// 		// 	cuCtxSynchronize();
	// 		// 	count++;
	// 		// 	now = steady_clock::now();

	// 		// 	if (tend <= steady_clock::now())
	// 		// 		break;
	// 		// }

	// 		this_thread::sleep_for(milliseconds(timer));
	// 		stop = true;

	// 		results[j] = 0;

	// 		for (int i = 0; i < (j + 1); i++)
	// 		{
	// 			th[i].join();
	// 			results[j] += fps[i];
	// 		}

	// 		// d = now - tstart;
	// 		// results[j] = d.count() / count * 1000000;

	// 		printf("%6.3lfus\n", results[j]);

	// 		// if (smCount != 68)
	// 	}

	// 	// ctx[0]->release();
	// 	cout << "Saving results\n";
	// 	writeToFile("interference", smCount, results);
	// 	cout << "-------------------------------------------------------------------------------\n\n";
	// }

	// void dummThread(int index, MyContext* ctx, bool* stop, double* fps)
	// {
	// 	int count = 0;
	// 	ctx->select();

	// 	auto tstart = steady_clock::now();

	// 	while (!*stop)
	// 	{
	// 		mdi[index]->forward(ini[index]);
	// 		cuCtxSynchronize();
	// 		count++;
	// 	}

	// 	auto tend = steady_clock::now();
	// 	duration<double> d = tend - tstart;
	// 	*fps = count / d.count();
	// 	ctx->release();
}