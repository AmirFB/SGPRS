# include <sqnl.hpp>

# include <cnt.hpp>

using namespace FGPRS;

void MySequential::setLevel(int level) { _maxLevel = level; }

void MySequential::addContainer(shared_ptr<MyContainer> container)
{
	containers.push_back(container);
}

void MySequential::copyOperations(string parentName, MyContainer& container, int level)
{
	MyContainer::copyOperations(parentName, container, level);

	_maxLevel = container._maxLevel;
	containers.insert(containers.end(), container.containers.begin(), container.containers.end());
}

Tensor MySequential::schedule(Tensor input, int level)
{
	if (level > _maxLevel || containers.size() == 0)
		return MyContainer::schedule(input, level);

	for (auto cont : containers)
		input = cont->schedule(input, level);

	return input;
}

Tensor MySequential::analyze(int warmup, int repeat, Tensor input, int index, int level)
{
	if (level > _maxLevel || containers.size() == 0)
		return MyContainer::analyze(warmup, repeat, input, index, level);

	for (auto cont : containers)
		input = cont->analyze(warmup, repeat, input, index, level);

	return input;
}

double MySequential::assignExecutionTime(int level, int contextIndex, double executionTimeStack)
{
	double tempStack = 0, elapsedTime = 0, tempElapsed;

	if (level > _maxLevel || containers.size() == 0)
		return MyContainer::assignExecutionTime(level, contextIndex, executionTimeStack);

	for (auto cont : containers)
	{
		tempStack = cont->assignExecutionTime(level, contextIndex, executionTimeStack);
		tempElapsed = tempStack - executionTimeStack;
		elapsedTime += tempElapsed;
		executionTimeStack = tempStack;
	}

	level--;
	regulatedExecutionTime[level] = elapsedTime;
	return executionTimeStack;
}

double MySequential::assignDeadline(double quota, int level, int contextIndex, double deadlineStack)
{
	if (level > _maxLevel || containers.size() == 0)
		return MyContainer::assignDeadline(quota, level, contextIndex, deadlineStack);

	level--;

	for (auto cont : containers)
		deadlineStack = cont->assignDeadline((cont->regulatedExecutionTime[level] / regulatedExecutionTime[level]) * quota, level + 1, contextIndex, deadlineStack);

	return deadlineStack;
}