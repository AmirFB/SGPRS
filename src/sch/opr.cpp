# include <main.hpp>

# include <opr.hpp>

# include <ctxd.hpp>
# include <schd.hpp>

# include <torch/torch.h>
# include <c10/cuda/CUDAStream.h>
# include <cuda_runtime_api.h>
# include <nvToolsExt.h>
# include <c10/cuda/CUDACachingAllocator.h>

# include <chrono>
# include <iostream>
# include <unistd.h>
# include <future>
# include <cmath>

using namespace std;
using namespace chrono;
using namespace FGPRS;

double Operation::exceptionThreshold = 0.15;

string Operation::getName() { return _name; }
string Operation::getFullName() { return _fullName; }

void Operation::setName(string name)
{
	_name = name;
	_fullName = name;
	strcpy(_cName, _fullName.c_str());
}

void Operation::setParentName(string parentName)
{
	if (_lastParentName != parentName)
	{
		_fullName = parentName + "->" + _fullName;
		_lastParentName = parentName;
		strcpy(_cName, _fullName.c_str());
	}
}

Tensor Operation::analyze(int warmup, int repeat, Tensor input, int index)
{
#ifdef ENABLE_NVTX_PROFILING
	auto mainId = nvtxRangeStartA(_fullName.c_str());
#endif

	NoGradGuard no_grad;
	Tensor output;
	bool first = true;
	steady_clock::time_point t1, t2, tStart, tEnd, tNow;
	int countIsolated, countOccupied;

	predictability = 0;
	isolatedScalability = 0;
	occupiedScalability = 0;

	_parent->analyzeLogger->info("{}:", _fullName);
	contextData.clear();

#ifdef ENABLE_NVTX_PROFILING
	auto rangeId = nvtxRangeStartA("warmup");
#endif

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		auto sm = Scheduler::smOptions[i];
		auto ctx = Scheduler::selectContextByIndex(i);
		ctx->select();

		_stream = at::cuda::getStreamFromPool(highPriority, ctx->index);
		_stream.select();
		at::cuda::setCurrentCUDAStream(_stream);

		for (int i = 0; i < warmup; i++)
		{
#ifdef ENABLE_NVTX_PROFILING
			auto innerId = nvtxRangeStartA(("warmup " + _fullName + " " + to_string(i)).c_str());
#endif

			output = sequential->forward(input);
			_stream.synchronize();

#ifdef ENABLE_NVTX_PROFILING
			nvtxRangeEnd(innerId);
#endif
		}

		ctx->release();
	}

#ifdef ENABLE_NVTX_PROFILING
	nvtxRangeEnd(rangeId);
#endif

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		auto sm = Scheduler::smOptions[i];
		auto ctx = Scheduler::selectContextByIndex(i);
		ctx->select();
		_stream = at::cuda::getStreamFromPool(highPriority, ctx->index);
		at::cuda::setCurrentCUDAStream(_stream);

#ifdef ENABLE_NVTX_PROFILING
		rangeId = nvtxRangeStartA((to_string(sm) + " SMs(iso)").c_str());
#endif

		tStart = steady_clock::now();

		for (int i = 0; i < repeat; i++)
		{
#ifdef ENABLE_NVTX_PROFILING
			auto innerId = nvtxRangeStartA(("iso " + _fullName + " " + to_string(i)).c_str());
#endif

			output = sequential->forward(input);
			// _stream.synchronize();
			cudaStreamSynchronize(_stream.stream());

#ifdef ENABLE_NVTX_PROFILING
			nvtxRangeEnd(innerId);
#endif
		}

		// cuCtxSynchronize();
		tNow = steady_clock::now();

#ifdef ENABLE_NVTX_PROFILING
		nvtxRangeEnd(rangeId);
#endif

		duration<double> d1 = tNow - tStart;

		ctx->release();
		Scheduler::startDummy(ctx, index);
		usleep(1000);
		ctx->select();

		countOccupied = 0;
		// tStart = steady_clock::now();
		// tEnd = steady_clock::now() + milliseconds(repeat);

#ifdef ENABLE_NVTX_PROFILING
		rangeId = nvtxRangeStartA((to_string(sm) + " SMs(occ)").c_str());
#endif

		at::cuda::setCurrentCUDAStream(_stream);
		tStart = steady_clock::now();

		for (int i = 0; i < repeat; i++)
		{
#ifdef ENABLE_NVTX_PROFILING
			auto innerId = nvtxRangeStartA(("occ " + _fullName + " " + to_string(i)).c_str());
#endif

			output = sequential->forward(input);
			// _stream.synchronize();
			cudaStreamSynchronize(_stream.stream());

#ifdef ENABLE_NVTX_PROFILING
			nvtxRangeEnd(innerId);
#endif
		}

		// cuCtxSynchronize();

		tNow = steady_clock::now();
		_stream.release();

#ifdef ENABLE_NVTX_PROFILING
		nvtxRangeEnd(rangeId);
#endif

		duration<double> d2 = tNow - tStart;

		Scheduler::stopDummy();
		ctx->release();
		torch::cuda::synchronize();

		contextData.push_back(ContextData(ctx, d1.count() / repeat * 1000000, d2.count() / repeat * 1000000));
		_parent->analyzeLogger->info("\t{}\t{:.0f}us, {:.0f}us",
			ctx->smCount, contextData.back().isolatedExecutionTime, contextData.back().occupiedExecutionTime);

		if (!first)
		{
			contextData.back().isolatedExecutionTime =
				min(contextData.back().isolatedExecutionTime, contextData[contextData.size() - 2].isolatedExecutionTime);
			contextData.back().occupiedExecutionTime =
				min(contextData.back().occupiedExecutionTime, contextData[contextData.size() - 2].occupiedExecutionTime);
		}

		predictability += 1 - (contextData.back().occupiedExecutionTime - contextData.back().isolatedExecutionTime) / contextData.back().occupiedExecutionTime;

		if (first)
		{
			first = false;
			continue;
		}

		double desired, isolatedGain, occupiedGain;

		desired = (double)contextData.back().smCount / contextData.end()[-2].smCount;
		isolatedGain = contextData.end()[-2].isolatedExecutionTime / contextData.back().isolatedExecutionTime;
		occupiedGain = contextData.end()[-2].occupiedExecutionTime / contextData.back().occupiedExecutionTime;

		isolatedScalability += max((isolatedGain - 1) / (desired - 1), 0.0);
		occupiedScalability += max((occupiedGain - 1) / (desired - 1), 0.0);
		// cout << "\t" << ctx->smCount << "\t" << contextData.back().isolatedExecutionTime << "us"
		// 	<< ", " << contextData.back().occupiedExecutionTime << "us";
	}

	predictability /= 4;
	isolatedScalability /= 3;
	occupiedScalability /= 3;
	// cout << endl
	// 	<< "Params: " << predictability << "\t" << isolatedScalability << "\t" << occupiedScalability << endl
	// 	<< endl;
	_parent->analyzeLogger->info("Params: {:.2f}\t{:.2f}\t{:.2f}", predictability, isolatedScalability, occupiedScalability);

#ifdef ENABLE_NVTX_PROFILING
	nvtxRangeEnd(mainId);
#endif

	return output;
}

vector<Tensor> Operation::analyzeSIMO(int warmup, int repeat, Tensor input, int index)
{
#ifdef ENABLE_NVTX_PROFILING
	auto mainId = nvtxRangeStartA(_fullName.c_str());
#endif

	NoGradGuard no_grad;
	vector<Tensor> output;
	bool first = true;
	steady_clock::time_point t1, t2, tStart, tEnd, tNow;
	int countIsolated, countOccupied;

	predictability = 0;
	isolatedScalability = 0;
	occupiedScalability = 0;

	_parent->analyzeLogger->info("{}:", _fullName);
	contextData.clear();

#ifdef ENABLE_NVTX_PROFILING
	auto rangeId = nvtxRangeStartA("warmup");
#endif

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		auto sm = Scheduler::smOptions[i];
		auto ctx = Scheduler::selectContextByIndex(i);
		ctx->select();

		_stream = at::cuda::getStreamFromPool(highPriority, ctx->index);
		_stream.select();
		at::cuda::setCurrentCUDAStream(_stream);

		for (int i = 0; i < warmup; i++)
		{
#ifdef ENABLE_NVTX_PROFILING
			auto innerId = nvtxRangeStartA(("warmup " + _fullName + " " + to_string(i)).c_str());
#endif

			output = container->forwardSIMO(input);
			_stream.synchronize();

#ifdef ENABLE_NVTX_PROFILING
			nvtxRangeEnd(innerId);
#endif
		}

		ctx->release();
	}

#ifdef ENABLE_NVTX_PROFILING
	nvtxRangeEnd(rangeId);
#endif

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		auto sm = Scheduler::smOptions[i];
		auto ctx = Scheduler::selectContextByIndex(i);
		ctx->select();
		_stream = at::cuda::getStreamFromPool(highPriority, ctx->index);
		at::cuda::setCurrentCUDAStream(_stream);

#ifdef ENABLE_NVTX_PROFILING
		rangeId = nvtxRangeStartA((to_string(sm) + " SMs(iso)").c_str());
#endif

		tStart = steady_clock::now();

		for (int i = 0; i < repeat; i++)
		{
#ifdef ENABLE_NVTX_PROFILING
			auto innerId = nvtxRangeStartA(("iso " + _fullName + " " + to_string(i)).c_str());
#endif

			output = container->forwardSIMO(input);
			// _stream.synchronize();
			cudaStreamSynchronize(_stream.stream());

#ifdef ENABLE_NVTX_PROFILING
			nvtxRangeEnd(innerId);
#endif
		}

		// cuCtxSynchronize();
		tNow = steady_clock::now();

#ifdef ENABLE_NVTX_PROFILING
		nvtxRangeEnd(rangeId);
#endif

		duration<double> d1 = tNow - tStart;

		ctx->release();
		Scheduler::startDummy(ctx, index);
		usleep(1000);
		ctx->select();

		countOccupied = 0;
		// tStart = steady_clock::now();
		// tEnd = steady_clock::now() + milliseconds(repeat);

#ifdef ENABLE_NVTX_PROFILING
		rangeId = nvtxRangeStartA((to_string(sm) + " SMs(occ)").c_str());
#endif

		at::cuda::setCurrentCUDAStream(_stream);
		tStart = steady_clock::now();

		for (int i = 0; i < repeat; i++)
		{
#ifdef ENABLE_NVTX_PROFILING
			auto innerId = nvtxRangeStartA(("occ " + _fullName + " " + to_string(i)).c_str());
#endif

			output = container->forwardSIMO(input);
			// _stream.synchronize();
			cudaStreamSynchronize(_stream.stream());

#ifdef ENABLE_NVTX_PROFILING
			nvtxRangeEnd(innerId);
#endif
		}

		// cuCtxSynchronize();

		tNow = steady_clock::now();
		_stream.release();

#ifdef ENABLE_NVTX_PROFILING
		nvtxRangeEnd(rangeId);
#endif

		duration<double> d2 = tNow - tStart;

		Scheduler::stopDummy();
		ctx->release();
		torch::cuda::synchronize();

		contextData.push_back(ContextData(ctx, d1.count() / repeat * 1000000, d2.count() / repeat * 1000000));
		_parent->analyzeLogger->info("\t{}\t{:.0f}us, {:.0f}us",
			ctx->smCount, contextData.back().isolatedExecutionTime, contextData.back().occupiedExecutionTime);

		if (!first)
		{
			contextData.back().isolatedExecutionTime =
				min(contextData.back().isolatedExecutionTime, contextData[contextData.size() - 2].isolatedExecutionTime);
			contextData.back().occupiedExecutionTime =
				min(contextData.back().occupiedExecutionTime, contextData[contextData.size() - 2].occupiedExecutionTime);
		}

		predictability += 1 - (contextData.back().occupiedExecutionTime - contextData.back().isolatedExecutionTime) / contextData.back().occupiedExecutionTime;

		if (first)
		{
			first = false;
			continue;
		}

		double desired, isolatedGain, occupiedGain;

		desired = (double)contextData.back().smCount / contextData.end()[-2].smCount;
		isolatedGain = contextData.end()[-2].isolatedExecutionTime / contextData.back().isolatedExecutionTime;
		occupiedGain = contextData.end()[-2].occupiedExecutionTime / contextData.back().occupiedExecutionTime;

		isolatedScalability += max((isolatedGain - 1) / (desired - 1), 0.0);
		occupiedScalability += max((occupiedGain - 1) / (desired - 1), 0.0);
		// cout << "\t" << ctx->smCount << "\t" << contextData.back().isolatedExecutionTime << "us"
		// 	<< ", " << contextData.back().occupiedExecutionTime << "us";
	}

	predictability /= 4;
	isolatedScalability /= 3;
	occupiedScalability /= 3;
	// cout << endl
	// 	<< "Params: " << predictability << "\t" << isolatedScalability << "\t" << occupiedScalability << endl
	// 	<< endl;
	_parent->analyzeLogger->info("Params: {:.2f}\t{:.2f}\t{:.2f}", predictability, isolatedScalability, occupiedScalability);

#ifdef ENABLE_NVTX_PROFILING
	nvtxRangeEnd(mainId);
#endif

	return output;
}

vector<Tensor> analyzeSIMO2(int warmup, int repeat, Tensor input, int index)
{
	// NoGradGuard no_grad;
	// vector<Tensor> output;
	// bool first = true;
	// steady_clock::time_point t1, t2, tStart, tEnd, tNow;
	// int countIsolated, countOccupied;

	// predictability = 0;
	// isolatedScalability = 0;
	// occupiedScalability = 0;

	// _parent->analyzeLogger->info("{}:", _fullName);

	// MyContext::selectDefault();
	// contextData.clear();

	// tStart = steady_clock::now();
	// tEnd = tStart + milliseconds(warmup);

	// for (int i = 0; i < warmup; i++)
	// 	output = container->forwardSIMO(input);

	// for (auto sm : Scheduler::smOptions)
	// {
	// 	auto ctx = Scheduler::selectContext(sm);
	// 	ctx->select();

	// 	tStart = steady_clock::now();

	// 	for (int i = 0; i < repeat; i++)
	// 		output = container->forwardSIMO(input);

	// 	tNow = steady_clock::now();

	// 	duration<double> d1 = tNow - tStart;

	// 	ctx->release();
	// 	Scheduler::startDummy(ctx, index);
	// 	// usleep(1000);
	// 	ctx->select();

	// 	countOccupied = 0;
	// 	tStart = steady_clock::now();
	// 	tEnd = tStart + milliseconds(repeat);

	// 	tStart = steady_clock::now();

	// 	for (int i = 0; i < repeat; i++)
	// 		output = container->forwardSIMO(input);

	// 	tNow = steady_clock::now();

	// 	duration<double> d2 = tNow - tStart;

	// 	Scheduler::stopDummy();
	// 	ctx->release();

	// 	contextData.push_back(ContextData(ctx, d1.count() / repeat * 1000000, d2.count() / repeat * 1000000));
	// 	_parent->analyzeLogger->info("\t{}\t{:.0f}us, {:.0f}us",
	// 		ctx->smCount, contextData.back().isolatedExecutionTime, contextData.back().occupiedExecutionTime);

	// 	if (!first)
	// 	{
	// 		contextData.back().isolatedExecutionTime =
	// 			min(contextData.back().isolatedExecutionTime, contextData[contextData.size() - 2].isolatedExecutionTime);
	// 		contextData.back().occupiedExecutionTime =
	// 			min(contextData.back().occupiedExecutionTime, contextData[contextData.size() - 2].occupiedExecutionTime);
	// 	}

	// 	predictability += 1 - (contextData.back().occupiedExecutionTime - contextData.back().isolatedExecutionTime) / contextData.back().occupiedExecutionTime;

	// 	if (first)
	// 	{
	// 		first = false;
	// 		continue;
	// 	}

	// 	double desired, isolatedGain, occupiedGain;

	// 	desired = (double)contextData.back().smCount / contextData.end()[-2].smCount;
	// 	isolatedGain = contextData.end()[-2].isolatedExecutionTime / contextData.back().isolatedExecutionTime;
	// 	occupiedGain = contextData.end()[-2].occupiedExecutionTime / contextData.back().occupiedExecutionTime;

	// 	isolatedScalability += max((isolatedGain - 1) / (desired - 1), 0.0);
	// 	occupiedScalability += max((occupiedGain - 1) / (desired - 1), 0.0);
	// }

	// predictability /= 4;
	// isolatedScalability /= 3;
	// occupiedScalability /= 3;

	// _parent->analyzeLogger->info("Params: {:.2f}\t{:.2f}\t{:.2f}", predictability, isolatedScalability, occupiedScalability);

	// return output;
}

void thrdFunction(Operation* operation, Tensor* input)
{
	// _chosenContext = Scheduler::getMinimalContext(this);
	// // _chosenContext = Scheduler::getFastestContext(this);
	// _chosenContext->queueOperation(operation);

	// *input = operation->sequential->forward(*input);

	// _chosenContext->dequeueOperation(operation);

	*input = operation->scheduleSync(*input);
}

void Operation::start(Tensor* input)
{
	_output = new Tensor();
	*_output = *input;

	_th = thread(thrdFunction, this, _output);
}

Tensor Operation::getResult()
{
	if (!_isException)
		_th.join();

	return *_output;
}
mutex mLock;

Tensor Operation::runSync(Tensor input)
{
	c10::cuda::CUDAStream::setContextIndex(_chosenContext->index);

	do
	{
		// _stream = at::cuda::getStreamFromPool(highPriority || _parent->delayed, _chosenContext->index);
		_stream = at::cuda::getStreamFromPool(highPriority, _chosenContext->index);
	}
	while (_stream.isBusy());

	_stream.select();
	at::cuda::setCurrentCUDAStream(_stream);

	// cout << "Let's see: " << ((int)_stream.device_index() == _chosenContext->index) << endl;

#ifdef ENABLE_NVTX_PROFILING
	auto id = nvtxRangeStartA(_fullName.c_str());
#endif

	// cout << "R " << _parent->_name << ": " << _fullName << "\tindex: " << (int)_stream.device_index() << ", " << (int)_stream.id() << endl;
	auto output = sequential->forward(input);
	// sequential->reset();
	// input.reset();
	// cout << "F " << _parent->_name << ": " << _fullName << "\tindex: " << (int)_stream.device_index() << ", " << (int)_stream.id() << endl;
	// if (highPriority || _parent->delayed)
	_stream.synchronize();
	// cudaStreamSynchronize(_stream.stream());
	// cout << "S " << _parent->_name << ": " << _fullName << "\tindex: " << (int)_stream.device_index() << ", " << (int)_stream.id() << endl;
	_stream.release();

#ifdef ENABLE_NVTX_PROFILING
	nvtxRangeEnd(id);
#endif

	return output;
	// return input + input;
}

vector<Tensor> Operation::runSIMOSync(Tensor input)
{
	_stream = at::cuda::getStreamFromPool(highPriority, _chosenContext->index);
	at::cuda::setCurrentCUDAStream(_stream);

#ifdef ENABLE_NVTX_PROFILING
	auto id = nvtxRangeStartA((_parent->_name + "-->" + _fullName).c_str());
#endif

	auto output = container->forwardSIMO(input);

#ifdef ENABLE_NVTX_PROFILING
	nvtxRangeEnd(id);
#endif

	return output;
}

void Operation::startSchedule(Tensor* input)
{
	auto now = steady_clock::now();

	if (false)//occupiedScalability < exceptionThreshold)
	{
		// _isException = true;
		// _chosenContext = Scheduler::selectDefaultContext();

		// _chosenContext->select();
		// _chosenContext->lock();
		// _chosenContext->queueOperation(this);

		// runSync(*input);

		// _chosenContext->unlock();
		// _chosenContext->release();
		// _chosenContext->dequeueOperation(operation);
	}

	else
	{
		_isException = false;
		start(input);
	}
}

Tensor Operation::scheduleSync(Tensor input)
{
	startTime = steady_clock::now();

	if (false)//occupiedScalability < exceptionThreshold)
	{
		_isException = true;
		_chosenContext = Scheduler::selectDefaultContext();
		_chosenContext->queueOperation(this);
		_chosenContext->lock(this);
	}

	else
	{
		_isException = false;

		if (!isLatest && !highPriority && !Scheduler::anyEmptyContext())
		{
			// cout << "name: " << _parent->_name << "->" << _fullName << "\t(Let's SLEEP!!!)" << endl;
			this_thread::sleep_until(earliestTime);
		}

		// if (!highPriority && Scheduler::anyEmptyContext() && (steady_clock::now() < earliestTime))
		{
			// cout << "name: " << _parent->_name << "->" << _fullName << "\t(There is empty so let\'s go!)" << endl;
			// _parent->scheduleLogger->info("Skipped: {}", _fullName.c_str());

			// cout << "earliestTime: " << duration_cast<microseconds>(earliestTime - steady_clock::now()).count() << endl;
			// cout << "earliestTime: " << earliestTime.time_since_epoch().count() << endl;
			// cout << "now: " << steady_clock::now().time_since_epoch().count() << endl;
			// cout << endl;
		}

		_chosenContext = Scheduler::getMinimalContext(this);
		// _chosenContext = Scheduler::getFastestContext(this);
	}

	_chosenContext->select();

	// _parent->scheduleLogger->info("Start  {}: {} SMs -> {}",
	// 	_fullName.c_str(), _chosenContext->smCount, queueCount);

	// _parent->scheduleLogger->info("Start  {}: {} SMs -> {}\t\t\tMemory Before: {}GB",
	// 	_fullName.c_str(), _chosenContext->smCount, _chosenContext->queue.size(), Scheduler::getFreeMemoryGB());

	auto output = runSync(input);
	// input.reset();
	// c10::cuda::CUDACachingAllocator::emptyCache();
	// torch::cuda::synchronize();

	// _parent->scheduleLogger->info("{}", (int)contextData[_chosenContext->index].occupiedExecutionTime);
	// _parent->scheduleLogger->info(
	// 	"End    {}: {} ({})-> {} + {} = {} ({})",
	// 	_fullName.c_str(),
	// 	(int)contextData[_chosenContext->index - 1].occupiedExecutionTime,
	// 	duration_cast<microseconds>(finishTime - startTime).count(),
	// 	duration_cast<microseconds>(steady_clock::now() - startTime).count(),
	// 	duration_cast<microseconds>(absoluteDeadline - steady_clock::now()).count(),
	// 	duration_cast<microseconds>(absoluteDeadline - startTime).count(),
	// 	(long)relativeDeadline[2]);

	// _parent->scheduleLogger->info(
	// 	"End    {}: {} -> {} + {} = {} ({})\t\t\tMemory  After: {}GB",
	// 	_fullName.c_str(),
	// 	(int)contextData[_chosenContext->index].occupiedExecutionTime,
	// 	duration_cast<microseconds>(steady_clock::now() - startTime).count(),
	// 	duration_cast<microseconds>(absoluteDeadline - steady_clock::now()).count(),
	// 	duration_cast<microseconds>(absoluteDeadline - startTime).count(),
	// 	(long)relativeDeadline[2], Scheduler::getFreeMemoryGB());

	if (steady_clock::now() > absoluteDeadline)
		_parent->delayed = true;

	else
		_parent->delayed = false;

	mLock.lock();
	_chosenContext->dequeueOperation(this);
	_chosenContext->unlock(this);
	mLock.unlock();

	return output;
}

vector<Tensor> Operation::scheduleSIMOSync(Tensor input)
{
	startTime = steady_clock::now();

	if (false)//occupiedScalability < exceptionThreshold)
	{
		_isException = true;
		_chosenContext = Scheduler::selectDefaultContext();
		_chosenContext->queueOperation(this);
		_chosenContext->lock(this);
	}

	else
	{
		_isException = false;
		_chosenContext = Scheduler::getMinimalContext(this);
		// _chosenContext = Scheduler::getFastestContext(this);
	}

	_chosenContext->select();

	_parent->scheduleLogger->info("Start  {}: {} SMs -> {}",
		_fullName.c_str(), _chosenContext->smCount, _chosenContext->queue.size());

	// _parent->scheduleLogger->info("Start  {}: {} SMs -> {}\t\t\tMemory Before: {}GB",
	// 	_fullName.c_str(), _chosenContext->smCount, _chosenContext->queue.size(), Scheduler::getFreeMemoryGB());

	auto output = runSIMOSync(input);

	_parent->scheduleLogger->info(
		"End    {}: {} -> {} + {} = {} ({})",
		_fullName.c_str(),
		(int)contextData[_chosenContext->index].occupiedExecutionTime,
		duration_cast<microseconds>(steady_clock::now() - startTime).count(),
		duration_cast<microseconds>(absoluteDeadline - steady_clock::now()).count(),
		duration_cast<microseconds>(absoluteDeadline - startTime).count(),
		(long)relativeDeadline[2]);

	// _parent->scheduleLogger->info(
	// 	"End    {}: {} -> {} + {} = {} ({})\t\t\tMemory  After: {}GB",
	// 	_fullName.c_str(),
	// 	(int)contextData[_chosenContext->index].occupiedExecutionTime,
	// 	duration_cast<microseconds>(steady_clock::now() - startTime).count(),
	// 	duration_cast<microseconds>(absoluteDeadline - steady_clock::now()).count(),
	// 	duration_cast<microseconds>(absoluteDeadline - startTime).count(),
	// 	(long)relativeDeadline[2], Scheduler::getFreeMemoryGB());

	_chosenContext->release();
	_chosenContext->dequeueOperation(this);
	_chosenContext->unlock(this);

	return output;
}

double Operation::getRegulatedExecutionTime(int contextIndex)
{
	return contextData[contextIndex].occupiedExecutionTime;// *max(1 - occupiedScalability, 0.25);
}

void Operation::setAbsoluteDeadline(int level, steady_clock::time_point start, int bias)
{
	absoluteDeadline = start + microseconds((int)stackedDeadline[level - 1] + bias);
	earliestTime = absoluteDeadline - microseconds((int)relativeDeadline[level - 1] * 2);
	// cout << level << endl;
	// cout << getFullName() << "->" << stackedDeadline[level - 1] << endl;

}