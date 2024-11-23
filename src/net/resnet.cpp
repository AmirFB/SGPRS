# include <memory>
# include <stdexcept>
# include <vector>
# include <string>

# include <torch/torch.h>

# include <ctxd.hpp>
# include <cnt.hpp>
# include <sqnl.hpp>
# include <schd.hpp>

using namespace std;
using namespace FGPRS;

Conv2dOptions
create_conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
	int64_t stride = 1, int64_t padding = 0, int64_t groups = 1,
	int64_t dilation = 1, bool bias = false)
{
	Conv2dOptions conv_options =
		Conv2dOptions(in_planes, out_planes, kerner_size)
		.stride(stride)
		.padding(padding)
		.bias(bias)
		.groups(groups)
		.dilation(dilation);

	return conv_options;
}

Conv2dOptions create_conv3x3_options(int64_t in_planes,
	int64_t out_planes,
	int64_t stride = 1,
	int64_t groups = 1,
	int64_t dilation = 1)
{
	Conv2dOptions conv_options = create_conv_options(
		in_planes, out_planes, /*kerner_size = */ 3, stride,
		/*padding = */ dilation, groups, /*dilation = */ dilation, false);
	return conv_options;
}

Conv2dOptions create_conv1x1_options(int64_t in_planes,
	int64_t out_planes,
	int64_t stride = 1)
{
	Conv2dOptions conv_options = create_conv_options(
		in_planes, out_planes,
		/*kerner_size = */ 1, stride,
		/*padding = */ 0, /*groups = */ 1, /*dilation = */ 1, false);
	return conv_options;
}

struct BasicBlock : public MyContainer
{
	BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
		MySequential downsample = MySequential(),
		int64_t groups = 1, int64_t base_width = 64,
		int64_t dilation = 1)
	{
		if ((groups != 1) || (base_width != 64))
		{
			throw invalid_argument{
				"BasicBlock only supports groups=1 and base_width=64" };
		}
		if (dilation > 1)
		{
			throw invalid_argument{
				"Dilation > 1 not supported in BasicBlock" };
		}
		m_conv1 =
			register_module("conv1", Conv2d{ create_conv3x3_options(
				inplanes, planes, stride) });
		m_bn1 = register_module("bn1", BatchNorm2d{ planes });
		m_relu = register_module("relu", ReLU{ true });
		m_conv2 = register_module(
			"conv2", Conv2d{ create_conv3x3_options(planes, planes) });
		m_bn2 = register_module("bn2", BatchNorm2d{ planes });
		if (!downsample.is_empty())
		{
			m_downsample = register_module("downsample", downsample);
		}
		m_stride = stride;
	}

	static const int64_t m_expansion = 1;

	Conv2d m_conv1{ nullptr }, m_conv2{ nullptr };
	BatchNorm2d m_bn1{ nullptr }, m_bn2{ nullptr };
	ReLU m_relu{ nullptr };
	MySequential m_downsample = MySequential();

	shared_ptr<Operation> o_conv1, o_conv2, o_bn1, o_bn2, o_relu1, o_relu2, o_downsample;

	int64_t m_stride;

	Tensor forward(Tensor x)
	{
		Tensor identity = x;

		Tensor out = m_conv1->forward(x);
		out = m_bn1->forward(out);
		out = m_relu->forward(out);

		out = m_conv2->forward(out);
		out = m_bn2->forward(out);

		if (!m_downsample->is_empty())
			identity = m_downsample->forward(x);

		out += identity;
		out = m_relu->forward(out);

		return out;
	}

	Tensor schedule(Tensor input, int level) override
	{
		Tensor output;

		if (!m_downsample->is_empty())
			o_downsample->startSchedule(&input);

		output = o_conv1->scheduleSync(input);
		output = o_bn1->scheduleSync(output);
		output = o_relu1->scheduleSync(output);

		output = o_conv2->scheduleSync(output);
		output = o_bn2->scheduleSync(output);

		if (!m_downsample->is_empty())
			input = o_downsample->getResult();

		output += input;
		output = o_relu2->scheduleSync(output);

		return output;
	}

	void assignOperations()
	{
		o_conv1 = addOperation(this, "conv1", m_conv1.ptr());
		o_bn1 = addOperation(this, "bn1", m_bn1.ptr());
		o_relu1 = addOperation(this, "relu", m_relu.ptr());

		o_conv2 = addOperation(this, "conv2", m_conv2.ptr());
		o_bn2 = addOperation(this, "bn2", m_bn2.ptr());

		if (!m_downsample->is_empty())
			o_downsample = addOperation(this, "downsample", m_downsample.ptr());

		o_relu2 = addOperation(this, "relu", m_relu.ptr());
	}

	Tensor analyze(int warmup, int repeat, Tensor input, int index, int level)
	{
		Tensor output;

		output = o_conv1->analyze(warmup, repeat, input, index);
		output = o_bn1->analyze(warmup, repeat, output, index);
		output = o_relu1->analyze(warmup, repeat, output, index);
		output = o_conv2->analyze(warmup, repeat, output, index);

		if (!m_downsample->is_empty())
			input = o_downsample->analyze(warmup, repeat, input, index);

		output += input;
		output = o_relu2->analyze(warmup, repeat, output, index);

		return output;
	}

	double assignExecutionTime(int level, int contextIndex, double executionTimeStack) override
	{
		double tempStack = 0;
		level -= 1;
		cout << "Or here?\n";

		contextData[level].resize(Scheduler::contextCount);

		for (int i = 0; i < Scheduler::contextCount; i++)
		{
			contextData[level][i] = ContextData(Scheduler::selectContextByIndex(i));

			contextData[level][i].stackExecutionTime(o_conv1->contextData[i]);
			contextData[level][i].stackExecutionTime(o_bn1->contextData[i]);
			contextData[level][i].stackExecutionTime(o_relu1->contextData[i]);
			contextData[level][i].stackExecutionTime(o_conv2->contextData[i]);
			contextData[level][i].stackExecutionTime(o_relu2->contextData[i]);
		}

		tempStack += o_conv1->getRegulatedExecutionTime(contextIndex);
		tempStack += o_bn1->getRegulatedExecutionTime(contextIndex);
		tempStack += o_relu1->getRegulatedExecutionTime(contextIndex);
		tempStack += o_conv2->getRegulatedExecutionTime(contextIndex);
		tempStack += o_relu2->getRegulatedExecutionTime(contextIndex);

		regulatedExecutionTime[level] = tempStack;
		return executionTimeStack + tempStack;
	}

	double assignDeadline(double quota, int level, int contextIndex, double deadlineStack) override
	{
		double usedDeadline = 0;

		level -= 1;

		o_conv1->relativeDeadline[level] = o_conv1->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		usedDeadline += o_conv1->relativeDeadline[level];
		deadlineStack += o_conv1->relativeDeadline[level];
		o_conv1->stackedDeadline[level] = deadlineStack;

		// o_conv1->_parent->deadlineLogger->info("encoder->aspp->aspppooling-: {:.0f}", o_aspppooling->relativeDeadline[level]);

		// cout << o_conv1->getFullName() << ": " << o_conv1->relativeDeadline[level] << "-->" << o_conv1->stackedDeadline[level] << endl;

		o_bn1->relativeDeadline[level] = o_bn1->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		usedDeadline += o_bn1->relativeDeadline[level];
		deadlineStack += o_bn1->relativeDeadline[level];
		o_bn1->stackedDeadline[level] = deadlineStack;

		// cout << o_bn1->getFullName() << ": " << o_bn1->relativeDeadline[level] << "-->" << o_bn1->stackedDeadline[level] << endl;

		o_relu1->relativeDeadline[level] = o_relu1->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		usedDeadline += o_relu1->relativeDeadline[level];
		deadlineStack += o_relu1->relativeDeadline[level];
		o_relu1->stackedDeadline[level] = deadlineStack;

		// cout << o_relu1->getFullName() << ": " << o_relu1->relativeDeadline[level] << "-->" << o_relu1->stackedDeadline[level] << endl;

		o_conv2->relativeDeadline[level] = o_conv2->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		usedDeadline += o_conv2->relativeDeadline[level];
		deadlineStack += o_conv2->relativeDeadline[level];
		o_conv2->stackedDeadline[level] = deadlineStack;

		// cout << o_conv2->getFullName() << ": " << o_conv2->relativeDeadline[level] << "-->" << o_conv2->stackedDeadline[level] << endl;

		if (!m_downsample->is_empty())
		{
			o_downsample->relativeDeadline[level] = quota;
			o_downsample->stackedDeadline[level] = deadlineStack;

			// cout << o_downsample->getFullName() << ": " << o_downsample->relativeDeadline[level] << "-->" << o_downsample->stackedDeadline[level] << endl;
		}

		o_relu2->relativeDeadline[level] = o_relu2->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		usedDeadline += o_relu2->relativeDeadline[level];
		deadlineStack += o_relu2->relativeDeadline[level];
		o_relu2->stackedDeadline[level] = deadlineStack;

		// cout << o_relu2->getFullName() << ": " << o_relu2->relativeDeadline[level] << "-->" << o_relu2->stackedDeadline[level] << endl;

		return deadlineStack;
	}
};

struct Bottleneck : public MyContainer
{
	Bottleneck(int64_t inplanes, int64_t planes, int64_t stride = 1,
		MySequential downsample = MySequential(),
		int64_t groups = 1, int64_t base_width = 64,
		int64_t dilation = 1)
	{
		int64_t width = planes * (base_width / 64) * groups;

		m_conv1 = register_module(
			"conv1",
			Conv2d{ create_conv1x1_options(inplanes, width) });
		m_bn1 = register_module("bn1", BatchNorm2d{ width });
		m_conv2 = register_module("conv2",
			Conv2d{ create_conv3x3_options(
				width, width, stride, groups, dilation) });
		m_bn2 = register_module("bn2", BatchNorm2d{ width });
		m_conv3 =
			register_module("conv3", Conv2d{ create_conv1x1_options(
				width, planes * m_expansion) });
		m_bn3 = register_module("bn3",
			BatchNorm2d{ planes * m_expansion });
		m_relu = register_module("relu", ReLU{ true });
		if (!downsample->is_empty())
		{
			m_downsample = register_module("downsample", downsample);
		}
		m_stride = stride;
	}

	static const int64_t m_expansion = 4;

	Conv2d m_conv1{ nullptr }, m_conv2{ nullptr }, m_conv3{ nullptr };
	BatchNorm2d m_bn1{ nullptr }, m_bn2{ nullptr }, m_bn3{ nullptr };
	ReLU m_relu{ nullptr };
	MySequential m_downsample = MySequential();

	shared_ptr<Operation> o_conv1, o_conv2, o_conv3, o_bn1, o_bn2, o_bn3, o_relu1, o_relu2, o_relu3, o_downsample;

	int64_t m_stride;

	Tensor forward(Tensor x)
	{
		Tensor identity = x;

		Tensor out = m_conv1->forward(x);
		out = m_bn1->forward(out);
		out = m_relu->forward(out);

		out = m_conv2->forward(out);
		out = m_bn2->forward(out);
		out = m_relu->forward(out);

		out = m_conv3->forward(out);
		out = m_bn3->forward(out);

		if (!m_downsample->is_empty())
		{
			identity = m_downsample->forward(x);
		}

		out += identity;
		out = m_relu->forward(out);

		return out;
	}

	void assignOperations()
	{
		o_conv1 = addOperation(this, "conv1", m_conv1.ptr());
		o_bn1 = addOperation(this, "bn1", m_bn1.ptr());
		o_relu1 = addOperation(this, "relu", m_relu.ptr());

		o_conv2 = addOperation(this, "conv2", m_conv2.ptr());
		o_bn2 = addOperation(this, "bn2", m_bn2.ptr());
		o_relu2 = addOperation(this, "relu", m_relu.ptr());

		o_conv3 = addOperation(this, "conv3", m_conv3.ptr());
		o_bn3 = addOperation(this, "bn3", m_bn3.ptr());

		o_conv2 = addOperation(this, "conv2", m_conv2.ptr());
		o_bn2 = addOperation(this, "bn2", m_bn2.ptr());

		if (!m_downsample->is_empty())
			o_downsample = addOperation(this, "downsample", m_downsample.ptr());

		o_relu3 = addOperation(this, "relu", m_relu.ptr());
	}

	Tensor analyze(int warmup, int repeat, Tensor input, int index, int level)
	{
		Tensor output;

		output = o_conv1->analyze(warmup, repeat, input, index);
		output = o_bn1->analyze(warmup, repeat, output, index);
		output = o_relu1->analyze(warmup, repeat, output, index);

		output = o_conv2->analyze(warmup, repeat, output, index);
		output = o_bn2->analyze(warmup, repeat, output, index);
		output = o_relu2->analyze(warmup, repeat, output, index);

		output = o_conv3->analyze(warmup, repeat, output, index);
		output = o_bn3->analyze(warmup, repeat, output, index);

		if (!m_downsample->is_empty())
			input = o_downsample->analyze(warmup, repeat, input, index);

		output += input;
		output = o_relu3->analyze(warmup, repeat, output, index);

		return output;
	}

	double assignExecutionTime(int level, int contextIndex, double executionTimeStack) override
	{
		double tempStack = 0;
		level -= 1;
		contextData[level].resize(Scheduler::contextCount);

		for (int i = 0; i < Scheduler::contextCount; i++)
		{
			contextData[level][i] = ContextData(Scheduler::selectContextByIndex(i));

			contextData[level][i].stackExecutionTime(o_conv1->contextData[i]);
			contextData[level][i].stackExecutionTime(o_bn1->contextData[i]);
			contextData[level][i].stackExecutionTime(o_relu1->contextData[i]);

			contextData[level][i].stackExecutionTime(o_conv2->contextData[i]);
			contextData[level][i].stackExecutionTime(o_bn2->contextData[i]);
			contextData[level][i].stackExecutionTime(o_relu2->contextData[i]);

			contextData[level][i].stackExecutionTime(o_conv3->contextData[i]);
			contextData[level][i].stackExecutionTime(o_bn3->contextData[i]);

			contextData[level][i].stackExecutionTime(o_relu3->contextData[i]);
		}

		tempStack += o_conv1->getRegulatedExecutionTime(contextIndex);
		tempStack += o_bn1->getRegulatedExecutionTime(contextIndex);
		tempStack += o_relu1->getRegulatedExecutionTime(contextIndex);

		tempStack += o_conv2->getRegulatedExecutionTime(contextIndex);
		tempStack += o_bn2->getRegulatedExecutionTime(contextIndex);
		tempStack += o_relu2->getRegulatedExecutionTime(contextIndex);

		tempStack += o_conv3->getRegulatedExecutionTime(contextIndex);
		tempStack += o_bn3->getRegulatedExecutionTime(contextIndex);

		tempStack += o_relu3->getRegulatedExecutionTime(contextIndex);

		regulatedExecutionTime[level] = tempStack;
		return executionTimeStack + tempStack;
	}
};

class XSequential : public Module
{
	AdaptiveAvgPool2d _avgpool{ nullptr };
	Linear _fc{ nullptr };
public:
	XSequential()
		: Module()
	{
	}

	XSequential(int64_t in_channels, int64_t num_classes)
		: Module()
	{
		_avgpool = register_module(
			"avgpool", AdaptiveAvgPool2d(
				AdaptiveAvgPool2dOptions({ 1, 1 })));
		_fc = register_module(
			"fc", Linear(in_channels, num_classes));

		_avgpool->eval();
		_avgpool->to(kCUDA);

		_fc->eval();
		_fc->to(kCUDA);
	}

	Tensor forward(Tensor x)
	{
		x = _avgpool->forward(x);
		x = flatten(x, 1);
		x = _fc->forward(x);
		return x;
	}
};

template <typename Block, typename = std::enable_if_t<is_base_of<MyContainer, Block>::value>>
struct ResNet : public MyContainer
{
	ResNet(const vector<int64_t> layers, int64_t num_classes = 1000,
		bool zero_init_residual = false, int64_t groups = 1,
		int64_t width_per_group = 64,
		vector<int64_t> replace_stride_with_dilation = {})
	{
		if (replace_stride_with_dilation.size() == 0)
		{
			// Each element in the tuple indicates if we should replace
			// the 2x2 stride with a dilated convolution instead.
			replace_stride_with_dilation = { false, false, false };
		}
		if (replace_stride_with_dilation.size() != 3)
		{
			throw invalid_argument{
				"replace_stride_with_dilation should be empty or have exactly "
					"three elements." };
		}

		m_groups = m_groups;
		m_base_width = width_per_group;

		m_conv1 = register_module(
			"conv1",
			Conv2d{ create_conv_options(
				/*in_planes = */ 3, /*out_planes = */ m_inplanes,
				/*kerner_size = */ 7, /*stride = */ 2, /*padding = */ 3,
				/*groups = */ 1, /*dilation = */ 1, /*bias = */ false) });

		m_bn1 = register_module("bn1", BatchNorm2d{ m_inplanes });
		m_relu = register_module("relu", ReLU{ true });
		m_maxpool = register_module(
			"maxpool",
			MaxPool2d{
			MaxPool2dOptions({ 3, 3 }).stride({ 2, 2 }).padding(
				{ 1, 1 }) });

		auto layer = _make_layer(64, layers.at(0));
		m_layer1 = register_module("layer1", layer);
		m_layer1.copyOperations("layer1", layer);

		layer = _make_layer(128, layers.at(1), 2, replace_stride_with_dilation.at(0));
		m_layer2 = register_module("layer2", layer);
		m_layer2.copyOperations("layer2", layer);

		layer = _make_layer(256, layers.at(2), 2, replace_stride_with_dilation.at(1));
		m_layer3 = register_module("layer3", layer);
		m_layer3.copyOperations("layer3", layer);

		layer = _make_layer(512, layers.at(3), 2, replace_stride_with_dilation.at(2), true);
		m_layer4 = register_module("layer4", layer);
		m_layer4.copyOperations("layer4", layer, true);

		m_avgpool = register_module(
			"avgpool", AdaptiveAvgPool2d(
				AdaptiveAvgPool2dOptions({ 1, 1 })));
		m_fc = register_module(
			"fc", Linear(512 * Block::m_expansion, num_classes));

		m_layerX = XSequential(512 * Block::m_expansion, num_classes);

		for (auto m : modules(false))
		{
			if (m->name() == "Conv2dImpl")
			{
				OrderedDict<string, Tensor>
					named_parameters = m->named_parameters(false);
				Tensor* ptr_w = named_parameters.find("weight");
				init::kaiming_normal_(*ptr_w, 0, kFanOut,
					kReLU);
			}
			else if ((m->name() == "BatchNormImpl") ||
				(m->name() == "GroupNormImpl"))
			{
				OrderedDict<string, Tensor>
					named_parameters = m->named_parameters(false);
				Tensor* ptr_w = named_parameters.find("weight");
				init::constant_(*ptr_w, 1.0);
				Tensor* ptr_b = named_parameters.find("bias");
				init::constant_(*ptr_b, 0.0);
			}
		}

		if (zero_init_residual)
		{
			for (auto m : modules(false))
			{
				if (m->name() == "Bottleneck")
				{
					OrderedDict<string, Tensor>
						named_parameters =
						m->named_modules()["bn3"]->named_parameters(false);
					Tensor* ptr_w = named_parameters.find("weight");
					init::constant_(*ptr_w, 0.0);
				}
				else if (m->name() == "BasicBlock")
				{
					OrderedDict<string, Tensor>
						named_parameters =
						m->named_modules()["bn2"]->named_parameters(false);
					Tensor* ptr_w = named_parameters.find("weight");
					init::constant_(*ptr_w, 0.0);
				}
			}
		}
	}

	int64_t m_inplanes = 64;
	int64_t m_dilation = 1;
	int64_t m_groups = 1;
	int64_t m_base_width = 64;

	Conv2d m_conv1{ nullptr };
	BatchNorm2d m_bn1{ nullptr };
	ReLU m_relu{ nullptr };
	MaxPool2d m_maxpool{ nullptr };
	MySequential m_layer1{ nullptr }, m_layer2{ nullptr },
		m_layer3{ nullptr }, m_layer4{ nullptr };
	AdaptiveAvgPool2d m_avgpool{ nullptr };
	Linear m_fc{ nullptr };
	MySequential m_layer0{ nullptr };
	XSequential m_layerX;

	// shared_ptr<Operation> o_conv1, o_bn1, o_relu, o_maxpool, o_layer1, o_layer2, o_layer3, o_layer4, o_avgpool, o_fc;
	// shared_ptr<Operation> o_layer0, o_layer1, o_layer2, o_layer3, o_layer4, o_avgpool, o_fc;
	shared_ptr<Operation> o_layer0, o_layer1, o_layer2, o_layer3, o_layer4, o_layerX;

	MySequential _make_layer(int64_t planes, int64_t blocks,
		int64_t stride = 1, bool dilate = false, bool highPriority = false)
	{
		MySequential downsample = MySequential();
		int64_t previous_dilation = m_dilation;

		if (dilate)
		{
			m_dilation *= stride;
			stride = 1;
		}

		if ((stride != 1) || (m_inplanes != planes * Block::m_expansion))
		{
			downsample = MySequential(
				Conv2d(create_conv1x1_options(
					m_inplanes, planes * Block::m_expansion, stride)),
				BatchNorm2d(planes * Block::m_expansion));
		}

		MySequential layers;

		auto block = Block(m_inplanes, planes, stride, downsample,
			m_groups, m_base_width, previous_dilation);
		layers->push_back(block);

		block.assignOperations();
		layers.copyOperations("block1", block, 2);
		layers.addOperation(this, "block1", make_shared<Block>(block), 2, highPriority);
		layers.addContainer(make_shared<Block>(block));

		m_inplanes = planes * Block::m_expansion;

		for (int64_t i = 0; i < blocks; i++)
		{
			block = Block(m_inplanes, planes, 1,
				MySequential(), m_groups,
				m_base_width, m_dilation);

			layers->push_back(block);
			block.assignOperations();
			layers.copyOperations("block" + to_string(i + 2), block, 2);
			layers.addOperation(this, "block" + to_string(i + 2), make_shared<Block>(block), 2, highPriority);
			layers.addContainer(make_shared<Block>(block));
		}

		layers.setLevel(1);
		return layers;
	}

	Tensor _forward_impl(Tensor x)
	{
		x = m_conv1->forward(x);
		x = m_bn1->forward(x);
		x = m_relu->forward(x);
		x = m_maxpool->forward(x);

		x = m_layer1->forward(x);
		x = m_layer2->forward(x);
		x = m_layer3->forward(x);
		x = m_layer4->forward(x);

		x = m_avgpool->forward(x);
		x = flatten(x, 1);
		x = m_fc->forward(x);

		return x;
	}

	Tensor forward(Tensor x) { return _forward_impl(x); }

	Tensor schedule(Tensor input, int level) override
	{
		Tensor output;

		output = o_layer0->scheduleSync(input);
		// input.reset();

		// input = o_conv1->scheduleSync(input);
		// input = o_bn1->scheduleSync(input);
		// input = o_relu->scheduleSync(input);
		// input = o_maxpool->scheduleSync(input);

		if (level == 3)
		{
			input = o_layer1->scheduleSync(output);
			// output.reset();
			output = o_layer2->scheduleSync(input);
			// input.reset();
			input = o_layer3->scheduleSync(output);
			// output.reset();
			output = o_layer4->scheduleSync(input);
			// input.reset();
		}

		else
		{
			input = m_layer1.schedule(input, level);
			input = m_layer2.schedule(input, level);
			input = m_layer3.schedule(input, level);
			input = m_layer4.schedule(input, level);
		}

		// input = o_avgpool->scheduleSync(input);
		// input = flatten(input, 1);
		// input = o_fc->scheduleSync(input);

		input = o_layerX->scheduleSync(output);
		// output.reset();

		return input;
	}

	void assignOperations()
	{
		m_layer0 = MySequential(m_conv1, m_bn1, m_relu, m_maxpool);
		o_layer0 = addOperation(this, "layer0", m_layer0.ptr(), 3);

		// o_conv1 = addOperation(this, "conv1", m_conv1.ptr());
		// o_bn1 = addOperation(this, "bn1", m_bn1.ptr());
		// o_relu = addOperation(this, "relu", m_relu.ptr());
		// o_maxpool = addOperation(this, "maxpool", m_maxpool.ptr());

		// copyOperations("layer1", m_layer1, 3);
		o_layer1 = addOperation(this, "layer1", m_layer1.ptr(), 3);

		// copyOperations("layer2", m_layer2, 3);
		o_layer2 = addOperation(this, "layer2", m_layer2.ptr(), 3);

		// copyOperations("layer3", m_layer3, 3);
		o_layer3 = addOperation(this, "layer3", m_layer3.ptr(), 3);

		// copyOperations("layer4", m_layer4, 3);
		o_layer4 = addOperation(this, "layer4", m_layer4.ptr(), 3, true);

		// m_layerX = MySequential(o_avgpool, o_fc);
		// o_layerX = addOperation(this, "layerX", m_layerX.ptr(), 3);

		// torch::nn::Sequential model{
		// 	torch::nn::Functional([&](torch::Tensor& x) {
		// 		x = m_avgpool->forward(x);
		// 		x = flatten(x, 1);
		// 		x = m_fc->forward(x);
		// 		return x;
		// 		})
		// };

		auto temp = make_shared<XSequential>(m_layerX);
		o_layerX = addOperation(this, "layerX", temp, 3, true, true);

		// o_avgpool = addOperation(this, "avgpool", m_avgpool.ptr(), 3, true);
		// o_fc = addOperation(this, "fc", m_fc.ptr(), 3, true);
	}

	Tensor analyze(int warmup, int repeat, Tensor input, int index, int level)
	{
		// input = o_conv1->analyze(warmup, repeat, input, index);
		// input = o_bn1->analyze(warmup, repeat, input, index);
		// input = o_relu->analyze(warmup, repeat, input, index);
		// input = o_maxpool->analyze(warmup, repeat, input, index);

		input = o_layer0->analyze(warmup, repeat, input, index);

		if (level == 3)
		{
			input = o_layer1->analyze(warmup, repeat, input, index);
			input = o_layer2->analyze(warmup, repeat, input, index);
			input = o_layer3->analyze(warmup, repeat, input, index);
			input = o_layer4->analyze(warmup, repeat, input, index);
		}

		else
		{
			input = m_layer1.analyze(warmup, repeat, input, index, level);
			input = m_layer2.analyze(warmup, repeat, input, index, level);
			input = m_layer3.analyze(warmup, repeat, input, index, level);
			input = m_layer4.analyze(warmup, repeat, input, index, level);
		}

		// input = o_avgpool->analyze(warmup, repeat, input, index);
		// input = flatten(input, 1);
		// input = o_fc->analyze(warmup, repeat, input, index);

		input = o_layerX->analyze(warmup, repeat, input, index);

		return input;
	}

	double assignExecutionTime(int level, int contextIndex, double executionTimeStack) override
	{
		double tempStack = 0;

		if (level != 1)
			return MyContainer::assignExecutionTime(level, contextIndex, executionTimeStack);

		level -= 1;

		contextData[level].resize(Scheduler::contextCount);

		for (int i = 0; i < Scheduler::contextCount; i++)
		{
			contextData[level][i] = ContextData(Scheduler::selectContextByIndex(i));

			// contextData[level][i].stackExecutionTime(o_conv1->contextData[i]);
			// contextData[level][i].stackExecutionTime(o_bn1->contextData[i]);
			// contextData[level][i].stackExecutionTime(o_relu->contextData[i]);
			// contextData[level][i].stackExecutionTime(o_maxpool->contextData[i]);

			contextData[level][i].stackExecutionTime(o_layer0->contextData[i]);
		}

		// tempStack += o_conv1->getRegulatedExecutionTime(contextIndex);
		// tempStack += o_bn1->getRegulatedExecutionTime(contextIndex);
		// tempStack += o_relu->getRegulatedExecutionTime(contextIndex);
		// tempStack += o_maxpool->getRegulatedExecutionTime(contextIndex);

		tempStack += o_layer0->getRegulatedExecutionTime(contextIndex);

		tempStack += m_layer1.assignExecutionTime(level + 1, contextIndex, 0);
		tempStack += m_layer2.assignExecutionTime(level + 1, contextIndex, 0);
		tempStack += m_layer3.assignExecutionTime(level + 1, contextIndex, 0);
		tempStack += m_layer4.assignExecutionTime(level + 1, contextIndex, 0);

		for (int i = 0; i < Scheduler::contextCount; i++)
		{
			// contextData[level][i] = ContextData(Scheduler::selectContextByIndex(i));

			// contextData[level][i].stackExecutionTime(o_avgpool->contextData[i]);
			// contextData[level][i].stackExecutionTime(o_fc->contextData[i]);

			contextData[level][i].stackExecutionTime(m_layer1.contextData[level][i]);
			contextData[level][i].stackExecutionTime(m_layer2.contextData[level][i]);
			contextData[level][i].stackExecutionTime(m_layer3.contextData[level][i]);
			contextData[level][i].stackExecutionTime(m_layer4.contextData[level][i]);

			contextData[level][i].stackExecutionTime(o_layerX->contextData[i]);

			// contextData[level][i].stackExecutionTime(o_avgpool->contextData[i]);
			// contextData[level][i].stackExecutionTime(o_fc->contextData[i]);
		}

		// tempStack += o_avgpool->getRegulatedExecutionTime(contextIndex);
		// tempStack += o_fc->getRegulatedExecutionTime(contextIndex);

		tempStack += o_layerX->getRegulatedExecutionTime(contextIndex);

		regulatedExecutionTime[level] = tempStack;
		return executionTimeStack + tempStack;
	}

	double assignDeadline(double quota, int level, int contextIndex, double deadlineStack) override
	{
		double usedDeadline = 0, tempStack, tempDeadline;

		if (level != 1)
			return MyContainer::assignDeadline(quota, level, contextIndex, deadlineStack);

		level -= 1;

		// o_conv1->relativeDeadline[level] = o_conv1->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		// usedDeadline += o_conv1->relativeDeadline[level];
		// deadlineStack += o_conv1->relativeDeadline[level];
		// o_conv1->stackedDeadline[level] = deadlineStack;

		// // cout << o_conv1->getFullName() << ": " << o_conv1->relativeDeadline[level] << "-->" << o_conv1->stackedDeadline[level] << endl;

		// o_bn1->relativeDeadline[level] = o_bn1->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		// usedDeadline += o_bn1->relativeDeadline[level];
		// deadlineStack += o_bn1->relativeDeadline[level];
		// o_bn1->stackedDeadline[level] = deadlineStack;

		// // cout << o_bn1->getFullName() << ": " << o_bn1->relativeDeadline[level] << "-->" << o_bn1->stackedDeadline[level] << endl;

		// o_relu->relativeDeadline[level] = o_relu->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		// usedDeadline += o_relu->relativeDeadline[level];
		// deadlineStack += o_relu->relativeDeadline[level];
		// o_relu->stackedDeadline[level] = deadlineStack;

		// // cout << o_relu->getFullName() << ": " << o_relu->relativeDeadline[level] << "-->" << o_relu->stackedDeadline[level] << endl;

		// o_maxpool->relativeDeadline[level] = o_maxpool->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		// usedDeadline += o_maxpool->relativeDeadline[level];
		// deadlineStack += o_maxpool->relativeDeadline[level];
		// o_maxpool->stackedDeadline[level] = deadlineStack;

		// // cout << o_maxpool->getFullName() << ": " << o_maxpool->relativeDeadline[level] << "-->" << o_maxpool->stackedDeadline[level] << endl;

		o_layer0->relativeDeadline[level] = o_layer0->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		usedDeadline += o_layer0->relativeDeadline[level];
		deadlineStack += o_layer0->relativeDeadline[level];
		o_layer0->stackedDeadline[level] = deadlineStack;

		tempStack = m_layer1.assignDeadline((m_layer1.regulatedExecutionTime[level] / regulatedExecutionTime[level]) * quota, level + 1, contextIndex, deadlineStack);
		tempDeadline = tempStack - deadlineStack;
		usedDeadline += tempDeadline;
		deadlineStack += tempDeadline;

		tempStack = m_layer2.assignDeadline((m_layer2.regulatedExecutionTime[level] / regulatedExecutionTime[level]) * quota, level + 1, contextIndex, deadlineStack);
		tempDeadline = tempStack - deadlineStack;
		usedDeadline += tempDeadline;
		deadlineStack += tempDeadline;

		tempStack = m_layer3.assignDeadline((m_layer3.regulatedExecutionTime[level] / regulatedExecutionTime[level]) * quota, level + 1, contextIndex, deadlineStack);
		tempDeadline = tempStack - deadlineStack;
		usedDeadline += tempDeadline;
		deadlineStack += tempDeadline;

		tempStack = m_layer4.assignDeadline((m_layer4.regulatedExecutionTime[level] / regulatedExecutionTime[level]) * quota, level + 1, contextIndex, deadlineStack);
		tempDeadline = tempStack - deadlineStack;
		usedDeadline += tempDeadline;
		deadlineStack += tempDeadline;

		// o_avgpool->relativeDeadline[level] = o_avgpool->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		// usedDeadline += o_avgpool->relativeDeadline[level];
		// deadlineStack += o_avgpool->relativeDeadline[level];
		// o_avgpool->stackedDeadline[level] = deadlineStack;

		// // cout << o_avgpool->getFullName() << ": " << o_avgpool->relativeDeadline[level] << "-->" << o_avgpool->stackedDeadline[level] << endl;

		// o_fc->relativeDeadline[level] = o_fc->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		// usedDeadline += o_fc->relativeDeadline[level];
		// deadlineStack += o_fc->relativeDeadline[level];
		// o_fc->stackedDeadline[level] = deadlineStack;

		// // cout << o_fc->getFullName() << ": " << o_fc->relativeDeadline[level] << "-->" << o_fc->stackedDeadline[level] << endl;

		o_layerX->relativeDeadline[level] = o_layerX->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
		usedDeadline += o_layerX->relativeDeadline[level];
		deadlineStack += o_layerX->relativeDeadline[level];
		o_layerX->stackedDeadline[level] = deadlineStack;

		return deadlineStack;
	}
};

template <class Block>
shared_ptr<ResNet<Block>>
_resnet(const vector<int64_t>& layers, int64_t num_classes = 1000,
	bool zero_init_residual = false, int64_t groups = 1,
	int64_t width_per_group = 64,
	const vector<int64_t>& replace_stride_with_dilation = {})
{
	shared_ptr<ResNet<Block>> model = make_shared<ResNet<Block>>(
		layers, num_classes, zero_init_residual, groups, width_per_group,
		replace_stride_with_dilation);
	return model;
}

shared_ptr<ResNet<BasicBlock>>
resnet18(int64_t num_classes = 1000, bool zero_init_residual = false,
	int64_t groups = 1, int64_t width_per_group = 64,
	vector<int64_t> replace_stride_with_dilation = {})
{
	const vector<int64_t> layers{ 2, 2, 2, 2 };
	shared_ptr<ResNet<BasicBlock>> model =
		_resnet<BasicBlock>(layers, num_classes, zero_init_residual, groups,
			width_per_group, replace_stride_with_dilation);
	return model;
}

shared_ptr<ResNet<BasicBlock>>
resnet34(int64_t num_classes = 1000, bool zero_init_residual = false,
	int64_t groups = 1, int64_t width_per_group = 64,
	vector<int64_t> replace_stride_with_dilation = {})
{
	const vector<int64_t> layers{ 3, 4, 6, 3 };
	shared_ptr<ResNet<BasicBlock>> model =
		_resnet<BasicBlock>(layers, num_classes, zero_init_residual, groups,
			width_per_group, replace_stride_with_dilation);
	return model;
}

shared_ptr<ResNet<Bottleneck>>
resnet50(int64_t num_classes = 1000, bool zero_init_residual = false,
	int64_t groups = 1, int64_t width_per_group = 64,
	vector<int64_t> replace_stride_with_dilation = {})
{
	const vector<int64_t> layers{ 3, 4, 6, 3 };
	shared_ptr<ResNet<Bottleneck>> model =
		_resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
			width_per_group, replace_stride_with_dilation);
	return model;
}

shared_ptr<ResNet<Bottleneck>>
resnet101(int64_t num_classes = 1000, bool zero_init_residual = false,
	int64_t groups = 1, int64_t width_per_group = 64,
	vector<int64_t> replace_stride_with_dilation = {})
{
	const vector<int64_t> layers{ 3, 4, 23, 3 };
	shared_ptr<ResNet<Bottleneck>> model =
		_resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
			width_per_group, replace_stride_with_dilation);
	return model;
}

shared_ptr<ResNet<Bottleneck>>
resnet152(int64_t num_classes = 1000, bool zero_init_residual = false,
	int64_t groups = 1, int64_t width_per_group = 64,
	vector<int64_t> replace_stride_with_dilation = {})
{
	const vector<int64_t> layers{ 3, 8, 36, 3 };
	shared_ptr<ResNet<Bottleneck>> model =
		_resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
			width_per_group, replace_stride_with_dilation);
	return model;
}

shared_ptr<ResNet<Bottleneck>>
resnext50_32x4d(int64_t num_classes = 1000, bool zero_init_residual = false,
	int64_t groups = 1, int64_t width_per_group = 64,
	vector<int64_t> replace_stride_with_dilation = {})
{
	groups = 32;
	width_per_group = 4;
	const vector<int64_t> layers{ 3, 4, 6, 3 };
	shared_ptr<ResNet<Bottleneck>> model =
		_resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
			width_per_group, replace_stride_with_dilation);
	return model;
}

shared_ptr<ResNet<Bottleneck>>
resnext101_32x8d(int64_t num_classes = 1000, bool zero_init_residual = false,
	int64_t groups = 1, int64_t width_per_group = 64,
	vector<int64_t> replace_stride_with_dilation = {})
{
	groups = 32;
	width_per_group = 8;
	const vector<int64_t> layers{ 3, 4, 23, 3 };
	shared_ptr<ResNet<Bottleneck>> model =
		_resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
			width_per_group, replace_stride_with_dilation);
	return model;
}

shared_ptr<ResNet<Bottleneck>>
wide_resnet50_2(int64_t num_classes = 1000, bool zero_init_residual = false,
	int64_t groups = 1, int64_t width_per_group = 64,
	vector<int64_t> replace_stride_with_dilation = {})
{
	width_per_group = 64 * 2;
	const vector<int64_t> layers{ 3, 4, 6, 3 };
	shared_ptr<ResNet<Bottleneck>> model =
		_resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
			width_per_group, replace_stride_with_dilation);
	return model;
}

shared_ptr<ResNet<Bottleneck>>
wide_resnet101_2(int64_t num_classes = 1000, bool zero_init_residual = false,
	int64_t groups = 1, int64_t width_per_group = 64,
	vector<int64_t> replace_stride_with_dilation = {})
{
	width_per_group = 64 * 2;
	const vector<int64_t> layers{ 3, 4, 23, 3 };
	shared_ptr<ResNet<Bottleneck>> model =
		_resnet<Bottleneck>(layers, num_classes, zero_init_residual, groups,
			width_per_group, replace_stride_with_dilation);
	return model;
}

template struct ResNet<BasicBlock>;
template struct ResNet<Bottleneck>;