# include <ctx.hpp>

# include <ctxd.hpp>
# include <schd.hpp>
# include <loop.hpp>

# include <iostream>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>

# include <torch/torch.h>
# include <c10/cuda/CUDAStream.h>

using namespace FGPRS;

using namespace std;
using namespace torch;

int MyContext::maxParallel;

MyContext::MyContext(unsigned smCount, int index, bool isDefault) :
	smCount(smCount), index(index), _default(isDefault),
	_pMutex(new mutex), _pQueueMutex(new mutex), //_lock(*_pMutex),
	cv(new condition_variable())
{
	// if (smCount <= 20)
	// 	maxParallel = 3;

	// else if (smCount < 40)
	// 	maxParallel = 3;

	// else if (smCount < 60)
	// 	maxParallel = 3;

	// else
	// maxParallel = 3;
}

bool MyContext::initialize()
{
	bool result = true;

	if (_default)
	{
		result &= cuInit(0) == CUDA_SUCCESS;
		result &= cuCtxGetCurrent(&_context) == CUDA_SUCCESS;
		// return result;
	}

	else
	{
		CUexecAffinityParam_v1 affinity;
		affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
		affinity.param.smCount.val = smCount;
		result &= cuCtxCreate_v3(&_context, &affinity, 1, 0, 0) == CUDA_SUCCESS;

		cuInit(0);
	}

	size_t temp;

	result &= cuCtxGetLimit(&temp, CU_LIMIT_PRINTF_FIFO_SIZE) == CUDA_SUCCESS;
	result &= cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, temp + (index << 8)) == CUDA_SUCCESS;

	result &= cuCtxGetLimit(&temp, CU_LIMIT_MALLOC_HEAP_SIZE) == CUDA_SUCCESS;
	result &= cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE,
		temp + ((Scheduler::contextCount + 1) << 16)) == CUDA_SUCCESS;

	return result;
}

// c10::cuda::CUDAStream* MyContext::select()
bool MyContext::select()
{
	// if (_default)
	// 	return MyContext::selectDefault();

	isBusy = true;
	c10::cuda::CUDAStream::setContextIndex(index);
	return cuCtxSetCurrent(_context) == CUDA_SUCCESS;
}

bool MyContext::selectDefault()
{
	return Scheduler::selectDefaultContext()->select();
}

bool MyContext::release()
{
	isBusy = false;
	// return selectDefault();
}

// void MyContext::lock()
// {
// 	cout << "Locking " << smCount << "\tcount: " << lockCount << endl;
// 	while (lockCount >= 2)
// 	{
// 		cv->wait(_lock);
// 		cout << "Notified " << smCount << "\tcount: " << lockCount << endl;
// 	}
// 	cout << "Locked " << smCount << endl;
// 	lockCount++;
// }

// void MyContext::unlock()
// {
// 	lockCount--;
// 	cv->notify_all();
// 	cout << "Unlocked " << smCount << endl;
// }

void MyContext::lock()
{
	unique_lock<mutex> lock(*_pMutex);

	while (lockCount >= maxParallel)
		cv->wait(lock);

	lockCount++;
}

void MyContext::lock(Operation* operation)
{
	lock();
	return;
	bool preCondition = !operation->isLatest && !operation->highPriority;
	unique_lock<mutex> lock(*_pMutex);

	if (preCondition)
		while (lockCount >= maxParallel && (!operation->isReady))
			cv->wait(lock);

	lockCount++;
	operation->running = true;
}

void MyContext::unlock()
{
	unique_lock<mutex> lock(*_pMutex);

	lockCount--;
	cv->notify_all();
}

void MyContext::unlock(Operation* operation)
{
	unlock();
	return;
	unique_lock<mutex> lock(*_pMutex);
	operation->running = false;
	operation->isReady = false;

	microseconds minSlack = microseconds::max(), tempSlack;
	auto now = steady_clock::now();
	Operation* minSlackOp = nullptr;

	for (auto op : queue)
	{
		if (op->running)
			continue;

		tempSlack = duration_cast<microseconds>(op->absoluteDeadline - now) - microseconds((int)op->contextData[index - 1].occupiedExecutionTime);

		if (tempSlack < minSlack)
		{
			minSlack = tempSlack;
			minSlackOp = op;
		}
	}

	// time_point earliestDeadline = now + microseconds(1000);

	// for (auto op : queue)
	// {
	// 	if (op->running)
	// 		continue;

	// 	if (op->absoluteDeadline < earliestDeadline)
	// 	{
	// 		earliestDeadline = op->absoluteDeadline;
	// 		minSlackOp = op;
	// 	}
	// }

	// if (minSlackOp != nullptr)
	// 	minSlackOp->isReady = true;

	lockCount--;
	cv->notify_all();
}

void MyContext::queueOperation(Operation* operation)
{
	_pQueueMutex->lock();
	_changed = true;
	queue.push_back(operation);
	_pQueueMutex->unlock();
}

void MyContext::dequeueOperation(Operation* operation)
{
	_pQueueMutex->lock();
	_changed = true;
	queue.erase(std::remove(queue.begin(), queue.end(), operation), queue.end());
	_pQueueMutex->unlock();
}

steady_clock::time_point MyContext::getFinishTime()
{
	if (queue.size() == 0)
		return _finishTime = steady_clock::now();

	// if (!_changed)
	// 	return _finishTime;

	_changed = false;
	double sum = 0;

	for (auto op : queue)
		sum += op->contextData[index - 1].occupiedExecutionTime;

	time_point<steady_clock> dummystart;

	if ((queue[0]->startTime + microseconds((int)queue[0]->contextData[index - 1].occupiedExecutionTime)) < steady_clock::now())
		dummystart = steady_clock::now();
	else
		dummystart = queue[0]->startTime;

	_finishTime = dummystart + microseconds((int)sum);
	return _finishTime;
}

bool MyContext::isEmpty()
{
	return queue.size() == 0;
}

bool MyContext::hasCapacity()
{
	return queue.size() < maxParallel;
}