# ifndef __SEQUENTIAL__
# define __SEQUENTIAL__

# include <ctxd.hpp>
# include <cnt.hpp>
# include <opr.hpp>

# include <torch/torch.h>

# include <chrono>
# include <future>
# include <stdio.h>
# include <memory>

using namespace torch;
using namespace torch::nn;

using namespace std;
using namespace chrono;

namespace FGPRS
{
	class MySequential : public MyContainer, public ModuleHolder<SequentialImpl>
	{
	public:
		using ModuleHolder<SequentialImpl>::ModuleHolder;

		MySequential() : ModuleHolder() {}
		MySequential(initializer_list<NamedAnyModule> named_modules) : ModuleHolder(make_shared<SequentialImpl>(move(named_modules))) {}

		void setLevel(int level);
		void addContainer(shared_ptr<MyContainer> container);
		void copyOperations(string parentName, MyContainer& container, int level = 1);
		Tensor schedule(Tensor input, int level) override;
		Tensor analyze(int warmup, int repeat, Tensor input, int index, int level) override;
		double assignExecutionTime(int level, int contextIndex, double executionTimeStack);
		double assignDeadline(double quota, int level, int contextIndex, double deadlineStack) override;
	};
}

# endif