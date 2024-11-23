# include <main.hpp>

# include <loop.hpp>

# include <cnt.hpp>
# include <schd.hpp>

# include <memory>
# include <chrono>
# include <thread>
# include <ctime>
# include <sys/time.h>
# include <pthread.h>
# include <vector>
# include <random>

# include <c10/cuda/CUDACachingAllocator.h>
# include <cuda_runtime_api.h>
# include <nvToolsExt.h>

using namespace FGPRS;

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace chrono;

bool first = true;

Loop::Loop(string name, shared_ptr<MyContainer> container, double frequency, int index)
	: _name(name), _container(container), _frequency(frequency), _period(1000000000 / frequency), _index(index)
{
}

void Loop::prepare()
{
	MyContext::selectDefault();
	_container->eval();
	_container->to(kCUDA);

	_container->initLoggers(_name);
}

void Loop::initialize(int deadlineContextIndex, Tensor dummyInput, SchedulerType type, int level)
{
	// auto stream = at::cuda::getStreamFromPool(false, 0);

	for (int j = 0; j < Scheduler::contextCount; j++)
	{
		auto ctx = Scheduler::selectContextByIndex(j);
		ctx->select();

		// if (type == PROPOSED_SCHEDULER && (j != (Scheduler::contextCount) && Scheduler::noDefault == true))
		// {

		for (int i = 0; i < 10; i++)
		{
			auto stream = at::cuda::getStreamFromPool(i % 2, ctx->index);
			at::cuda::setCurrentCUDAStream(stream);
			_container->forward(dummyInput);

			// if (j != Scheduler::contextCount)
			stream.synchronize();
			// cudaStreamSynchronize(stream.stream());
		}
		// }

		// else
		// 	for (int i = 0; i < 10; i++)
		// 		_container->forward(dummyInput);

		// cuCtxSynchronize();
		ctx->release();
	}

	_container->assignOperations();

	if (type == PROPOSED_SCHEDULER)
	{
		if (first)
		{
			// _container->clearAnalyzeLogger(_name);
#ifdef ENABLE_NVTX_PROFILING
			auto id = nvtxRangeStartA(_name.c_str());
#endif

			_container->analyze(5, 10, dummyInput, _index, level);

#ifdef ENABLE_NVTX_PROFILING
			nvtxRangeEnd(id);
#endif

			first = false;
		}

#ifdef ENABLE_NVTX_PROFILING
		auto id = nvtxRangeStartA(_name.c_str());
#endif

		// cout << "Analyzing " << _name << " ..." << endl;
		_container->analyze(10, 25, dummyInput, _index, level);
		// cout << "Analyzing " << _name << " done." << endl;

#ifdef ENABLE_NVTX_PROFILING
		nvtxRangeEnd(id);
#endif

		_container->assignExecutionTime(level, deadlineContextIndex, 0);
		_container->assignDeadline(_period / 1000 * 0.99, level, deadlineContextIndex, 0);
	}
}

void run(
	string name, shared_ptr<MyContainer> container, Tensor* input,
	double period, bool* stop, int level, int index,
	SchedulerType type, bool logIt, int timer, Loop* loop)
{
	NoGradGuard no_grad;
	int frame = 0;
	steady_clock::time_point startTime, nextTime;
	auto interval = nanoseconds((int)round(period));
	// container->clearScheduleLogger(name);
	MyContext* ctx;

	if (type == MPS_SCHEDULER || type == PMPS_SCHEDULER || type == PMPSO_SCHEDULER)
	{
		ctx = Scheduler::selectContextByIndex(index % Scheduler::contextCount + 1);
		ctx->select();
	}

	// c10::cuda::CUDAStream stream = at::cuda::getStreamFromPool(false, index);
	// at::cuda::setCurrentCUDAStream(stream);

	// cudaProfilerSetStringName(threadId, name.c_str());
	// pthread_t nativeHandle = myThread.native_handle();
	// pthread_setname_np(pthread_self(), name.c_str());

	// std::random_device rd;
	// std::mt19937 gen(rd());
	// std::uniform_int_distribution<> dis(0, interval);
	// int randomMicroseconds = dis(gen);

	// Add the random time interval to the start time
	srand(index);
	startTime = steady_clock::now() + std::chrono::milliseconds(rand() % (int)round(period / 1000000 * 2));
	auto endTime = startTime + milliseconds(timer) + interval / 10;

	// startTime = steady_clock::now();// +std::chrono::milliseconds(1);
	// auto endTime = startTime + milliseconds(1000);

	// startTime = steady_clock::now();
	nextTime = startTime + nanoseconds((int)round(period * 1));
	// nextTime = startTime + milliseconds(2);

	container->meets = 0;
	container->missed = 0;

	steady_clock::time_point dummyNow;

	std::this_thread::sleep_until(startTime);
	auto stream = at::cuda::getStreamFromPool(index % 2, 10);

	if (type != PROPOSED_SCHEDULER)
	{
		stream = at::cuda::getStreamFromPool(index % 2, ctx->index);
		while (stream.isBusy())
			stream = at::cuda::getStreamFromPool(index % 2, ctx->index);

		stream.select();
		at::cuda::setCurrentCUDAStream(stream);
	}

	// while (!*stop)
	while (true)
	{
#ifdef ENABLE_NVTX_PROFILING
		auto id = nvtxRangeStartA((name + " " + to_string(frame)).c_str());
#endif

		if (type == PROPOSED_SCHEDULER)
			container->setAbsoluteDeadline(level, nextTime, 0);

		nextTime += interval;
		frame++;

		// cout << name << " " << frame << endl;
		if (type == PROPOSED_SCHEDULER)
		{
			// auto out = container->schedule(*input, level);
			// out.reset();

			std::promise<torch::Tensor> promise;

			// Get the future associated with the promise
			std::future<torch::Tensor> future = promise.get_future();

			// Create and start the thread
			std::thread scheduleThread([container, &input, &promise, level]() {
				torch::Tensor result = container->schedule(*input, level); // Assuming level is defined
				promise.set_value(result); // Set the result for the future
				});

			// Wait for the thread to finish and get the result
			torch::Tensor out = future.get();

			// Join the thread
			scheduleThread.join();
		}

		else
		{
			container->forward(*input);
			// cuCtxSynchronize();
			stream.synchronize();
		}
		// cout << name << " " << frame << endl;
		dummyNow = steady_clock::now();

		if (dummyNow > nextTime)
		{
			if (logIt)
				container->scheduleLogger->info("Delayed : {}us", duration_cast<microseconds>(dummyNow - nextTime).count());

			while (dummyNow > (nextTime + interval))
			{
				nextTime += interval;
			}

			container->missed++;
			container->delayed = true;
		}

		else
		{
			if (logIt)
				container->scheduleLogger->info("Reserved: {}us", duration_cast<microseconds>(nextTime - dummyNow).count());

			container->meets++;
			container->delayed = false;
			// c10::cuda::CUDACachingAllocator::emptyCache();
		}

#ifdef ENABLE_NVTX_PROFILING
		nvtxRangeEnd(id);
#endif

		if (steady_clock::now() > endTime)
			break;

		// c10::cuda::CUDACachingAllocator::emptyCache();
		this_thread::sleep_until(nextTime);
		// cout << name << " " << frame << "\tstop: " << *stop << endl;
	}

	double expected, completed, met, missed;

	expected = ((ceil(1000000000. / period) * (timer / 1000.)));
	completed = frame;
	met = container->meets;
	missed = container->meets;

	expected = max(expected, completed);

	double comp, miss;

	comp = 100. * completed / expected;
	miss = 100. * (expected - met) / expected;

	string temp = name +
		"\n\tCompleted: " + to_string(comp) +
		"%" + "\tMissed   : " + to_string(miss) + "%\n";
	cout << temp;

	loop->totalCount = expected;
	loop->compCount = completed;
	loop->missCount = expected - met;
}

void Loop::start(Tensor* input, SchedulerType type, int level, bool logIt, int timer)
{
	_stop = false;

	_th = thread(run, _name, _container, input, _period, &_stop, level, _index, type, logIt, timer, this);
}

void Loop::stop()
{
	_stop = true;
}

void Loop::wait()
{
	_stop = true;
	_th.join();
}