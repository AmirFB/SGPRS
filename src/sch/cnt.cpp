# include <main.hpp>

# include <cnt.hpp>

# include <schd.hpp>
# include <ctxd.hpp>

# include <torch/torch.h>

# include <iostream>
# include <chrono>
# include <thread>
# include <unistd.h>
# include <filesystem>

# include <nvToolsExt.h>

using namespace torch;
using namespace torch::nn;

using namespace FGPRS;

using namespace std;
using namespace chrono;


void MyContainer::initLoggers(string name)
{
	_name = name;

	mkdir(("logs/" + name).c_str(), 0777);

	// remove(("logs/" + name + "/analyze.log").c_str());
	// remove(("logs/" + name + "/deadline.log").c_str());
	// remove(("logs/" + name + "/schedule.log").c_str());

	analyzeLogger = spdlog::basic_logger_mt(name + "_analyze", "logs/" + name + "/analyze.log");
	deadlineLogger = spdlog::basic_logger_mt(name + "_deadline", "logs/" + name + "/deadline.log");
	scheduleLogger = spdlog::basic_logger_mt(name + "_schedule", "logs/" + name + "/schedule.log");

	analyzeLogger->set_pattern("[%S.%f] %v");
	analyzeLogger->flush_on(spdlog::level::info);
	deadlineLogger->set_pattern("[%S.%f] %v");
	scheduleLogger->set_pattern("[%S.%f] %v");
}

void MyContainer::clearAnalyzeLogger(string name)
{
	remove(("logs/" + name + "/analyze.log").c_str());
	spdlog::drop(name + "_analyze");
	analyzeLogger = spdlog::basic_logger_mt(name + "_analyze", "logs/" + name + "/analyze.log");
	analyzeLogger->set_pattern("[%S.%f] %v");
}

void MyContainer::clearScheduleLogger(string name)
{
	remove(("logs/" + name + "/schedule.log").c_str());
	spdlog::drop(name + "_schedule");
	scheduleLogger = spdlog::basic_logger_mt(name + "_schedule", "logs/" + name + "/schedule.log");
	scheduleLogger->set_pattern("[%S.%f] %v");
}

vector<shared_ptr<Operation>> MyContainer::getOperations(int level)
{
	return operations[level - 1];
}

void MyContainer::copyOperations(string parentName, MyContainer& container, int level)
{
	for (auto op : container.getOperations(1))
	{
		op->setParentName(parentName);
		operations[0].push_back(op);
	}

	if (level != 2)
	{
		for (auto op : container.getOperations(2))
		{
			op->setParentName(parentName);
			operations[1].push_back(op);
		}
	}

	if (level == 1)
	{
		for (auto op : container.getOperations(3))
		{
			op->setParentName(parentName);
			operations[2].push_back(op);
		}
	}
}

void MyContainer::copyOperations(string parentName, MyContainer* container, int level)
{
	auto dummy = container->getOperations(1);

	for (auto op : container->getOperations(1))
	{
		op->setParentName(parentName);
		operations[0].push_back(op);
	}

	if (level != 2)
	{
		for (auto op : container->getOperations(2))
		{
			op->setParentName(parentName);
			operations[1].push_back(op);
		}
	}

	if (level == 1)
	{
		for (auto op : container->getOperations(3))
		{
			op->setParentName(parentName);
			operations[2].push_back(op);
		}
	}
}

Tensor MyContainer::schedule(Tensor input, int level)
{
	for (auto op : operations[level - 1])
		input = op->scheduleSync(input);

	return input;
}

void MyContainer::analyze(int warmup, int repeat, Tensor input, int index)
{
	for (int i = 1; i <= 3; i++)
		analyze(warmup, repeat, input, index, i);
}

Tensor MyContainer::analyze(int warmup, int repeat, Tensor input, int index, int level)
{
	for (auto op : operations[level - 1])
	{
#ifdef ENABLE_NVTX_PROFILING
		// auto id = nvtxRangeStartA(op->getFullName().c_str());
#endif

		input = op->analyze(warmup, repeat, input, index);

#ifdef ENABLE_NVTX_PROFILING
		// nvtxRangeEnd(id);
#endif
	}

	return input;
}

double MyContainer::assignExecutionTime(int level, int contextIndex, double executionTimeStack)
{
	double timeStack = 0;
	level -= 1;

	contextData[level].resize(Scheduler::contextCount);

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		contextData[level][i] = ContextData(Scheduler::selectContextByIndex(i));

		for (auto op : operations[level])
			contextData[level][i].stackExecutionTime(op->contextData[i]);
	}

	for (auto op : operations[level])
		timeStack += op->getRegulatedExecutionTime(contextIndex);

	regulatedExecutionTime[level] = timeStack;
	return executionTimeStack + timeStack;
}

double MyContainer::assignDeadline(double quota, int level, int contextIndex, double deadlineStack)
{
	level -= 1;

	for (auto op : operations[level])
	{
		op->relativeDeadline[level] = op->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		deadlineStack += op->relativeDeadline[level];
		op->stackedDeadline[level] = deadlineStack;
		deadlineLogger->info("{}: {:.0f}", op->getFullName().c_str(), op->relativeDeadline[level]);
		// cout << fixed << setprecision(0) << op->getFullName() << "(" << level << ", " << contextIndex << "):\n\t"
		// 	<< op->getRegulatedExecutionTime(contextIndex) << "-->" << regulatedExecutionTime[level] << endl;
		// cout << "\t"
		// 	<< op->relativeDeadline[level] << "-->" << op->stackedDeadline[level] << endl;
	}

	return deadlineStack;
}

void MyContainer::setAbsoluteDeadline(int level, steady_clock::time_point start, int bias)
{
	for (auto op : operations[level - 1])
		op->setAbsoluteDeadline(level, start, bias);
}