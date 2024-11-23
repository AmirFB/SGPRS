# ifndef __LOOP__
# define __LOOP__

# include <cnt.hpp>

# include <memory>

namespace FGPRS
{
	enum SchedulerType { PROPOSED_SCHEDULER, NOMPS_SCHEDULER, MPS_SCHEDULER, PMPS_SCHEDULER, PMPSO_SCHEDULER };

	class Loop
	{
	private:
		shared_ptr<MyContainer> _container;
		double _frequency, _period;
		bool _stop = false;
		thread _th;
		string _name;
		int _index;

	public:
		int totalCount, compCount, missCount;

		Loop() {}
		Loop(string name, shared_ptr<MyContainer> container, double _frequency, int index = -1);

		void prepare();
		void initialize(int deadlineContextIndex, Tensor dummyInput, SchedulerType type, int level);
		void start(Tensor* input, SchedulerType type, int level, bool logIt, int timer);
		void stop();
		void wait();

		int completed() { return _container->meets + _container->missed; }
		int met() { return _container->meets; }
	};
}

# endif