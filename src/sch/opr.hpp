# ifndef __OPERATION__
# define __OPERATION__

# include <cnt.hpp>
# include <ctxd.hpp>

# include <torch/torch.h>
# include <c10/cuda/CUDAStream.h>


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
	class MyContainer;

	class Operation
	{
	private:
		string _name, _fullName, _lastParentName;
		char _cName[100];
		thread _th;
		Tensor* _output;
		MyContext* _chosenContext;
		bool _isException;
		c10::cuda::CUDAStream _stream;
		static double exceptionThreshold;

	public:
		MyContainer* _parent;
		Sequential sequential;
		shared_ptr<MyContainer> container;
		double relativeDeadline[3], stackedDeadline[3];
		steady_clock::time_point absoluteDeadline;
		double isolatedScalability, occupiedScalability, predictability;
		vector<ContextData> contextData;
		steady_clock::time_point startTime, finishTime, earliestTime;
		bool highPriority = false, isLatest = false;
		int queueCount = 0;
		bool isReady = false;
		bool running = false;

		Operation() : _stream(c10::cuda::getStreamFromPool()) {}
		string getName();
		string getFullName();
		void setName(string name);
		void setParentName(string parentName);

		template <typename ModuleType>
		Operation(MyContainer* owner, string name, shared_ptr<ModuleType> module, bool isHighPriority = false, bool isLatest = false)
			: _parent(owner), _name(name), _fullName(name),
			sequential(Sequential(module)), _stream(c10::cuda::getStreamFromPool()),
			highPriority(isHighPriority), isLatest(isLatest)
		{
		}

		Operation(MyContainer* owner, string name, shared_ptr<MyContainer> module, bool dumm, bool isHighPriority = false, bool isLatest = false)
			: _parent(owner), _name(name), _fullName(name),
			container(module), _stream(c10::cuda::getStreamFromPool()),
			highPriority(isHighPriority), isLatest(isLatest)
		{
		}

		Tensor analyze(int warmup, int repeat, Tensor input, int index);
		vector<Tensor> analyzeSIMO(int warmup, int repeat, Tensor input, int index);

		void start(Tensor* input);
		Tensor getResult();
		// Tensor runSync(Tensor input, c10::cuda::CUDAStream* stream);
		// vector<Tensor> runSIMOSync(Tensor input, c10::cuda::CUDAStream* stream);
		Tensor runSync(Tensor input);
		vector<Tensor> runSIMOSync(Tensor input);

		void startSchedule(Tensor* input);
		Tensor scheduleSync(Tensor input);
		vector<Tensor> scheduleSIMOSync(Tensor input);
		Tensor scheduleMISOSync(vector<Tensor> input);

		double getRegulatedExecutionTime(int contextIndex);
		void setAbsoluteDeadline(int level, steady_clock::time_point start, int bias);
	};
}

# endif