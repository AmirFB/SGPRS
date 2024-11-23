# include <main.hpp>
# include <schd.hpp>

# include <iostream>
# include <thread>
# include <future>
# include <ranges>
# include <vector>
# include <unistd.h>
# include <mutex>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>
# include <nvToolsExt.h>

# include <torch/torch.h>

using namespace FGPRS;

using namespace std;

using namespace torch;
using namespace torch::nn;

int Scheduler::maxSmCount;
bool Scheduler::_stopDummy;
vector<int> Scheduler::smOptions;
MyContext* Scheduler::_contextPool;
MyContext* Scheduler::_defaultContext;
SchedulerType Scheduler::type;


vector<DummyContainer> Scheduler::dummyContainer;

// Sequential* Scheduler::_dummyModule;
// Tensor* Scheduler::_dummyInput;

int Scheduler::contextCount = 0;
bool Scheduler::noDefault;
MyContext dummyContext;
int dummyIndex = 0;

bool Scheduler::initialize(int options[], int size, SchedulerType type, bool noDefault)
{
	bool result = true;
	cudaDeviceProp prop;
	Scheduler::type = type;

	cudaGetDeviceProperties(&prop, 0);
	maxSmCount = prop.multiProcessorCount;
	Scheduler::noDefault = noDefault;
	contextCount = size + (noDefault ? 0 : 1);

	// if (type == PROPOSED_SCHEDULER)
	{
		// _dummyModule = new Sequential[dummyCount];
		// _dummyInput = new Tensor[dummyCount];

		// for (int i = 0; i < dummyCount; i++)
		// {
		// 	_dummyInput[i] = torch::randn({ 1, 16, 448, 448 }, kCUDA);
		// 	_dummyModule[i] = Sequential(
		// 		Conv2d(Conv2dOptions(16, 32, 3).stride(1).padding(3)),
		// 		BatchNorm2d(32),
		// 		ReLU(),
		// 		MaxPool2d(MaxPool2dOptions(2))
		// 	);

		// 	_dummyModule[i]->eval();
		// 	_dummyModule[i]->to(kCUDA);
		// }

		_contextPool = new MyContext[size + 1];
		smOptions = vector<int>(size + 1);

		smOptions[size] = maxSmCount;
		_contextPool[size] = MyContext(maxSmCount, 0, true);
		_defaultContext = &_contextPool[size];
		result &= _contextPool[size].initialize();

		if (!noDefault)
			auto dummy = c10::cuda::getStreamFromPool(false, _contextPool[size].index);
	}

	// else
	// {
	// 	smOptions = vector<int>(size);
	// 	_contextPool = new MyContext[size];

	// 	dummyContext = MyContext(maxSmCount, size, true);
	// 	_defaultContext = &dummyContext;
	// }

	for (int i = 0; i < size; i++)
	{
		smOptions[i] = (min(max(options[i], 2), maxSmCount));

		_contextPool[i] = MyContext(smOptions[i], i + 1);
		result &= _contextPool[i].initialize();

		if (type == PROPOSED_SCHEDULER)
			auto dummy = c10::cuda::getStreamFromPool(false, _contextPool[i].index);
	}

	return result;
}

MyContext* Scheduler::selectContext(int smCount)
{
	for (int i = 0; i < contextCount; i++)
		if (_contextPool[i].smCount >= smCount && !_contextPool[i].isBusy)
			return &_contextPool[i];

	return _defaultContext;
}

MyContext* Scheduler::selectContextByIndex(int index)
{
	return &_contextPool[index];
}

MyContext* Scheduler::selectDefaultContext()
{
	return &_contextPool[smOptions.size() - 1];
}

bool Scheduler::releaseContext(MyContext context)
{
	return context.release();
}

float Scheduler::getTotalMemoryMB()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return total / 1024. / 1024.;
}

float Scheduler::getTotalMemoryGB()
{
	return Scheduler::getTotalMemoryMB() / 1024;
}

float Scheduler::getFreeMemoryMB()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	return free / 1024. / 1024.;
}

float Scheduler::getFreeMemoryGB()
{
	return Scheduler::getFreeMemoryMB() / 1024;
}

float Scheduler::getMemoryPercentage()
{
	return Scheduler::getFreeMemoryMB() / Scheduler::getTotalMemoryMB() * 100;
}

void Scheduler::dummyFunction(MyContext* ctx, shared_ptr<MyContainer> mod, Tensor* in, c10::cuda::CUDAStream str)
{
	NoGradGuard no_grad;
	int counter = 0;
	ctx->select();
	c10::cuda::CUDAStream::setContextIndex(ctx->index);
	// auto str = c10::cuda::getStreamFromPool(false, ctx->index);

	// while (str.isBusy())
	// 	str = c10::cuda::getStreamFromPool(FLAGS_torch_jit_enable_new_executor, ctx->index);

	str.select();
	at::cuda::setCurrentCUDAStream(str);

#ifdef ENABLE_NVTX_PROFILING
	auto innerId = nvtxRangeStartA(("dummy " + mod->_name + " " + to_string(ctx->index)).c_str());
#endif

	while (!_stopDummy)
	{
		at::cuda::setCurrentCUDAStream(str);
		auto dummy = mod->forward(*in);
		counter++;
		// cuCtxSynchronize();
		str.synchronize();
	}

#ifdef ENABLE_NVTX_PROFILING
	nvtxRangeEnd(innerId);
#endif
	str.release();
	ctx->release();
}

future<void>* Scheduler::_th;

void Scheduler::startDummy(MyContext* ctx, int moduleIndex)
{
	int first = 1;
	int index = 0;
	int ctxIndex = contextCount - 1;
	Tensor dummyOutput;
	_stopDummy = false;
	MyContext* dummyContext;

	if (ctx->index == contextCount)
		ctxIndex--;

	_th = new future<void>[dummyContainer.size() - 1];

	for (auto dummy : dummyContainer)
	{
		// cout << "dummy " << dummy.index << endl;
		if (dummy.index == moduleIndex)
			continue;

		// cout << "dummy " << dummy.index << " " << ctxIndex << endl;
		ctxIndex = (ctxIndex + Scheduler::contextCount) % Scheduler::contextCount;
		// cout << "dummy " << dummy.index << " " << ctxIndex << endl;
		dummyContext = Scheduler::selectContextByIndex(ctxIndex);
		dummyContext->select();

		auto str = c10::cuda::getStreamFromPool(false, dummyContext->index);

		while (str.isBusy())
			str = c10::cuda::getStreamFromPool(FLAGS_torch_jit_enable_new_executor, ctx->index);

		str.select();
		at::cuda::setCurrentCUDAStream(str);
		dummyOutput = dummy.mod->forward(*dummy.in);
		str.synchronize();
		str.release();
		dummyContext->release();

		// cout << "Chosen context: " << dummyContext->index << endl;
		_th[index++] = async(launch::async, dummyFunction, dummyContext, dummy.mod, dummy.in, str);
		ctxIndex--;

		if (ctxIndex == (ctx->index - 1) && first)
		{
			// cout << "dummy " << dummy.index << " first " << first << endl;
			first--;
			ctxIndex--;
		}

		if (index == (dummyContainer.size() - 1))
			break;
	}

	// cout << "dummy " << moduleIndex << " started" << endl;
}

void Scheduler::stopDummy()
{
	_stopDummy = true;

	for (int i = 0; i < (dummyContainer.size() - 1); i++)
		_th[i].get();

	delete[] _th;
}

mutex globalMutex;

MyContext* Scheduler::getMinimalContext(Operation* operation)
{
	MyContext* ctx1 = NULL, * ctx2;
	globalMutex.lock();

	steady_clock::time_point earliest1 = steady_clock::now() + seconds(1), earliest2 = steady_clock::now() + seconds(1);
	steady_clock::time_point temp;

	for (int i = 0; i < contextCount; i++)
	{
		if (!_contextPool[i].isEmpty())
			continue;

		temp = _contextPool[i].getFinishTime() + microseconds((int)operation->contextData[i].occupiedExecutionTime);

		if (temp < operation->absoluteDeadline)
		{
			_contextPool[i].queueOperation(operation);
			operation->finishTime = _contextPool[i].getFinishTime();
			globalMutex.unlock();
			operation->queueCount = _contextPool[i].queue.size();
			_contextPool[i].lock(operation);

			return &_contextPool[i];
		}
	}

	for (int i = 0; i < contextCount; i++)
	{
		temp = _contextPool[i].getFinishTime() + microseconds((int)operation->contextData[i].occupiedExecutionTime);

		if (_contextPool[i].isEmpty() && temp < earliest1)
		{
			earliest1 = temp;
			ctx1 = &_contextPool[i];
		}

		if (temp < earliest2)
		{
			earliest2 = temp;
			ctx2 = &_contextPool[i];
		}
	}

	if (ctx1 != NULL)
	{
		ctx1->queueOperation(operation);
		operation->finishTime = ctx1->getFinishTime();
		globalMutex.unlock();
		operation->queueCount = ctx1->queue.size();
		ctx1->lock(operation);

		return ctx1;
	}


	else
	{
		ctx2->queueOperation(operation);
		operation->finishTime = ctx2->getFinishTime();
		globalMutex.unlock();
		operation->queueCount = ctx2->queue.size();
		ctx2->lock(operation);

		return ctx2;
	}
}

MyContext* Scheduler::getFastestContext(Operation* operation)
{
	MyContext* ctx;
	globalMutex.lock();

	steady_clock::time_point earliest = steady_clock::now() + seconds(1);
	steady_clock::time_point temp;

	for (int i = 0; i < contextCount; i++)
	{
		temp = _contextPool[i].getFinishTime() + microseconds((int)operation->contextData[i].occupiedExecutionTime);

		if (temp < earliest)
		{
			earliest = temp;
			ctx = &_contextPool[i];
		}
	}

	ctx->queueOperation(operation);
	operation->finishTime = ctx->getFinishTime();
	globalMutex.unlock();
	operation->queueCount = ctx->queue.size();
	ctx->lock(operation);

	return ctx;
}

bool Scheduler::anyEmptyContext()
{
	for (int i = 0; i < contextCount; i++)
		if (_contextPool[i].isEmpty())
			return true;

	return false;
}