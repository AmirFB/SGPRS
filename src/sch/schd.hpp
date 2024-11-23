# ifndef SCHD_H
# define SCHD_H

# include <ctx.hpp>

# include <loop.hpp>

# include <chrono>
# include <cstdint>
# include <vector>
# include <future>

# include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

namespace FGPRS
{
	struct DummyContainer
	{
		shared_ptr<MyContainer> mod;
		Tensor* in;
		int index;
	};

	class Scheduler
	{
	public:
		const int MAX_CONTEXT_COUNT = 48;
		static int maxSmCount;
		static vector<int> smOptions;
		static bool _stopDummy;
		static SchedulerType type;
		static int contextCount;
		static bool noDefault;
		static MyContext* _contextPool;
		static vector<DummyContainer> dummyContainer;

	private:
		static MyContext* _defaultContext;
		static future<void>* _th;

	public:
		static bool initialize(int[], int, SchedulerType type = NOMPS_SCHEDULER, bool noDefault = false);
		static MyContext* selectContext(int);
		static MyContext* selectContextByIndex(int index);
		static MyContext* selectDefaultContext();
		static bool releaseContext(MyContext);

		static vector<MyContext> getAllContexts();

		static float getTotalMemoryMB();
		static float getFreeMemoryMB();
		static float getTotalMemoryGB();
		static float getFreeMemoryGB();
		static float getMemoryPercentage();

		static void dummyFunction(MyContext* ctx, shared_ptr<MyContainer> mod, Tensor* in, c10::cuda::CUDAStream str);
		static void startDummy(MyContext* ctx, int index);
		static void stopDummy();

		static MyContext* getMinimalContext(Operation* operation);
		static MyContext* getFastestContext(Operation* operation);

		static bool anyEmptyContext();
	};
}

# endif