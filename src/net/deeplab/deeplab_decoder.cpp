# include <main.hpp>
# include "deeplab_decoder.hpp"

# include <schd.hpp>

using namespace torch;
using namespace nn;

MySequential ASPPConv(int in_channels, int out_channels, int dilation)
{
	return MySequential(
		Conv2d(conv_options(in_channels, out_channels, 3, 1, dilation, 1, false, dilation)),
		BatchNorm2d(BatchNorm2dOptions(out_channels)),
		ReLU()
	);
}

MySequential SeparableConv2d(int in_channels, int out_channels, int kernel_size, int stride,
	int padding, int dilation, bool bias)
{
	Conv2d dephtwise_conv = Conv2d(conv_options(in_channels, in_channels, kernel_size,
		stride, padding, 1, false, dilation));
	Conv2d pointwise_conv = Conv2d(conv_options(in_channels, out_channels, 1, 1, 0, 1, bias));
	return MySequential(dephtwise_conv, pointwise_conv);
};

MySequential ASPPSeparableConv(int in_channels, int out_channels, int dilation)
{
	MySequential seq = SeparableConv2d(in_channels, out_channels, 3, 1, dilation, dilation, false);
	seq->push_back(BatchNorm2d(BatchNorm2dOptions(out_channels)));
	seq->push_back(ReLU());
	return seq;
}

ASPPPoolingImpl::ASPPPoolingImpl(int in_channels, int out_channels)
{
	seq = MySequential(AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions(1)),
		Conv2d(conv_options(in_channels, out_channels, 1, 1, 0, 1, false)),
		BatchNorm2d(BatchNorm2dOptions(out_channels)),
		ReLU());
	register_module("seq", seq);
}

torch::Tensor ASPPPoolingImpl::forward(torch::Tensor x)
{
	auto residual(x.clone());
	x = seq->forward(x);
	x = at::upsample_bilinear2d(x, residual[0][0].sizes(), false);
	return x;
}

ASPPImpl::ASPPImpl(int in_channels, int out_channels, vector<int> atrous_rates, bool separable)
{
	m_modules.push_back(MySequential(Conv2d(conv_options(in_channels, out_channels, 1, 1, 0, 1, false)),
		BatchNorm2d(BatchNorm2dOptions(out_channels)),
		ReLU()));

	if (atrous_rates.size() != 3) cout << "size of atrous_rates must be 3";

	if (separable)
	{
		m_modules.push_back(ASPPSeparableConv(in_channels, out_channels, atrous_rates[0]));
		m_modules.push_back(ASPPSeparableConv(in_channels, out_channels, atrous_rates[1]));
		m_modules.push_back(ASPPSeparableConv(in_channels, out_channels, atrous_rates[2]));
	}
	else
	{
		m_modules.push_back(ASPPConv(in_channels, out_channels, atrous_rates[0]));
		m_modules.push_back(ASPPConv(in_channels, out_channels, atrous_rates[1]));
		m_modules.push_back(ASPPConv(in_channels, out_channels, atrous_rates[2]));
	}

	m_aspppooling = ASPPPooling(in_channels, out_channels);

	m_project = MySequential(
		Conv2d(conv_options(5 * out_channels, out_channels, 1, 1, 0, 1, false)),
		BatchNorm2d(BatchNorm2dOptions(out_channels)),
		ReLU(),
		Dropout(DropoutOptions(0.5)));

	for (int i = 0; i < m_modules.size(); i++)
		register_module("aspp" + to_string(i + 1), m_modules[i]);

	register_module("aspppooling", m_aspppooling);
	register_module("project", m_project);
}

torch::Tensor ASPPImpl::forward(torch::Tensor x)
{
	vector<torch::Tensor> res;

	for (int i = 0; i < m_modules.size(); i++)
		res.push_back(m_modules[i]->forward(x));

	res.push_back(m_aspppooling->forward(x));
	x = cat(res, 1);
	x = m_project->forward(x);
	return x;
}

void ASPPImpl::assignOperations(MyContainer* owner)
{
	this->owner = owner;

	for (int i = 0; i < m_modules.size(); i++)
		o_modules.push_back(addOperation(owner, "decoder->aspp->aspp" + to_string(i + 1), m_modules[i].ptr()));

	o_aspppooling = addOperation(owner, "decoder->aspp->aspppooling", m_aspppooling.ptr());
	o_project = addOperation(owner, "decoder->aspp->project", m_project.ptr());
}

Tensor ASPPImpl::schedule(Tensor input, int level)
{
	vector<Tensor> result;

	for (int i = 0; i < o_modules.size(); i++)
		o_modules[i]->startSchedule(&input);

	o_aspppooling->startSchedule(&input);

	for (int i = 0; i < o_modules.size(); i++)
		result.push_back(o_modules[i]->getResult());

	result.push_back(o_aspppooling->getResult());

	input = cat(result, 1);
	return o_project->scheduleSync(input);
}

Tensor ASPPImpl::analyze(int warmup, int repeat, Tensor input, int index, int level)
{
	vector<Tensor> result;

	for (int i = 0; i < o_modules.size(); i++)
		result.push_back(o_modules[i]->analyze(warmup, repeat, input, index));

	result.push_back(o_aspppooling->analyze(warmup, repeat, input, index));

	input = cat(result, 1);
	return o_project->analyze(warmup, repeat, input, index);
}

double ASPPImpl::assignExecutionTime(int level, int contextIndex, double executionTimeStack)
{
	double tempStack = 0;
	int maxIndex = 0;
	level -= 1;
	contextData[level].resize(Scheduler::contextCount);

	for (int i = 1; i < m_modules.size(); i++)
		if (o_modules[i]->getRegulatedExecutionTime(contextIndex) >
			o_modules[maxIndex]->getRegulatedExecutionTime(contextIndex))
			maxIndex = i;

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		contextData[level][i] = ContextData(Scheduler::selectContextByIndex(i));

		contextData[level][i].stackExecutionTime(o_modules[maxIndex]->contextData[i]);
		contextData[level][i].stackExecutionTime(o_project->contextData[i]);
	}

	tempStack += o_modules[maxIndex]->getRegulatedExecutionTime(contextIndex);
	tempStack += o_project->getRegulatedExecutionTime(contextIndex);

	regulatedExecutionTime[level] = tempStack;
	return executionTimeStack + tempStack;
}

double ASPPImpl::assignDeadline(double quota, int level, int contextIndex, double deadlineStack)
{
	double usedDeadline = 0, tempStack, tempDeadline;

	level -= 1;

	for (int i = 0; i < o_modules.size(); i++)
	{
		o_modules[i]->relativeDeadline[level] =
			(regulatedExecutionTime[level] - o_project->getRegulatedExecutionTime(contextIndex))
			/ regulatedExecutionTime[level] * quota;
		o_modules[i]->stackedDeadline[level] = deadlineStack + o_modules[i]->relativeDeadline[level];

		o_modules[i]->_parent->deadlineLogger->info("encoder->aspp->aspp{}-: {:.0f}", i + 1, o_modules[i]->relativeDeadline[level]);
	}

	o_aspppooling->relativeDeadline[level] =
		(regulatedExecutionTime[level] - o_project->getRegulatedExecutionTime(contextIndex))
		/ regulatedExecutionTime[level] * quota;
	usedDeadline += o_aspppooling->relativeDeadline[level];
	deadlineStack += o_aspppooling->relativeDeadline[level];
	o_aspppooling->stackedDeadline[level] = deadlineStack;

	o_aspppooling->_parent->deadlineLogger->info("encoder->aspp->aspppooling-: {:.0f}", o_aspppooling->relativeDeadline[level]);

	o_project->relativeDeadline[level] =
		o_project->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
	usedDeadline += o_project->relativeDeadline[level];
	deadlineStack += o_project->relativeDeadline[level];
	o_project->stackedDeadline[level] = deadlineStack;

	o_project->_parent->deadlineLogger->info("encoder->aspp->project-: {:.0f}", o_project->relativeDeadline[level]);

	return deadlineStack;
}

void ASPPImpl::setAbsoluteDeadline(int level, steady_clock::time_point start, int bias)
{
	for (int i = 0; i < o_modules.size(); i++)
		o_modules[i]->setAbsoluteDeadline(level, start, bias);

	o_aspppooling->setAbsoluteDeadline(level, start, bias);
	o_project->setAbsoluteDeadline(level, start, bias);
}

DeepLabV3DecoderImpl::DeepLabV3DecoderImpl(int in_channels, int out_channels, vector<int> atrous_rates)
{
	seq->push_back(ASPP(in_channels, out_channels, atrous_rates));
	seq->push_back(Conv2d(conv_options(out_channels, out_channels, 3, 1, 1, 1, false)));
	seq->push_back(BatchNorm2d(BatchNorm2dOptions(out_channels)));
	seq->push_back(ReLU());

	register_module("seq", seq);
}

torch::Tensor DeepLabV3DecoderImpl::forward(vector< torch::Tensor> x_list)
{
	auto x = seq->forward(x_list[x_list.size() - 1]);
	return x;
}

DeepLabV3PlusDecoderImpl::DeepLabV3PlusDecoderImpl(vector<int> encoder_channels, int out_channels,
	vector<int> atrous_rates, int output_stride)
{
	if (output_stride != 8 && output_stride != 16) cout << "Output stride should be 8 or 16";

	auto aspp_seq = SeparableConv2d(out_channels, out_channels, 3, 1, 1, 1, false);
	aspp_seq->push_back(BatchNorm2d(BatchNorm2dOptions(out_channels)));
	aspp_seq->push_back(ReLU());
	double scale_factor = double(output_stride / 4);

	int highres_in_channels = encoder_channels[encoder_channels.size() - 4];
	int highres_out_channels = 48; // proposed by authors of paper

	auto block2 = SeparableConv2d(highres_out_channels + out_channels, out_channels, 3, 1, 1, 1, false);
	block2->push_back(BatchNorm2d(BatchNorm2dOptions(out_channels)));
	block2->push_back(ReLU());

	m_aspp = register_module("aspp", ASPP(encoder_channels[encoder_channels.size() - 1], out_channels, atrous_rates, true));
	m_aspp_seq = register_module("aspp_seq", aspp_seq);
	m_block1 = register_module("block1", MySequential(
		Conv2d(conv_options(highres_in_channels, highres_out_channels, 1, 1, 0, 1, false)),
		BatchNorm2d(BatchNorm2dOptions(highres_out_channels)),
		ReLU()));
	m_up = register_module("up",
		Upsample(UpsampleOptions()
			.align_corners(true)
			.scale_factor(vector<double>({ scale_factor,scale_factor }))
			.mode(kBilinear)));
	m_block2 = register_module("block2", block2);
}

torch::Tensor DeepLabV3PlusDecoderImpl::forward(vector<torch::Tensor> x_list)
{
	auto aspp_features = m_aspp->forward(x_list[x_list.size() - 1]);
	aspp_features = m_aspp_seq->forward(aspp_features);
	aspp_features = m_up->forward(aspp_features);

	auto high_res_features = m_block1->forward(x_list[x_list.size() - 4]);
	auto concat_features = cat({ aspp_features, high_res_features }, 1);
	auto fused_features = m_block2->forward(concat_features);
	return fused_features;
}

void DeepLabV3PlusDecoderImpl::assignOperations(MyContainer* owner)
{
	owner = owner;

	m_aspp->assignOperations(owner);
	// copyOperations("aspp", (MyContainer*)&m_aspp, 3);
	o_aspp = addOperation(owner, "decoder->aspp", m_aspp.ptr(), 2);

	o_aspp_seq = addOperation(owner, "decoder->aspp_seq", m_aspp_seq.ptr(), true);
	o_up = addOperation(owner, "decoder->up", m_up.ptr(), true);
	o_block1 = addOperation(owner, "decoder->block1", m_block1.ptr());
	o_block2 = addOperation(owner, "decoder->block2", m_block2.ptr(), true);
}

Tensor DeepLabV3PlusDecoderImpl::scheduleMISO(vector<Tensor> inputs, int level)
{
	// cout << "inputs size: " << inputs.size() << "\n";
	// for (int i = 0; i < inputs.size(); i++)
	// 	cout << "input " << i << " size: " << inputs[i].sizes() << "\n";

	o_block1->startSchedule(&inputs[inputs.size() - 4]);

	auto aspp_features = m_aspp->schedule(inputs.back(), level);
	aspp_features = o_aspp_seq->scheduleSync(aspp_features);
	aspp_features = o_up->scheduleSync(aspp_features);

	auto high_res_features = o_block1->getResult();

	auto concat_features = cat({ aspp_features, high_res_features }, 1);
	return o_block2->scheduleSync(concat_features);
}

Tensor DeepLabV3PlusDecoderImpl::analyzeMISO(int warmup, int repeat, vector<Tensor> inputs, int index, int level)
{
	auto high_res_features = o_block1->analyze(warmup, repeat, inputs[inputs.size() - 4], index);
	auto aspp_features = m_aspp->analyze(warmup, repeat, inputs.back(), index, level);
	aspp_features = o_aspp_seq->analyze(warmup, repeat, aspp_features, index);
	aspp_features = o_up->analyze(warmup, repeat, aspp_features, index);

	auto concat_features = cat({ aspp_features, high_res_features }, 1);
	return o_block2->analyze(warmup, repeat, concat_features, index);
}

double DeepLabV3PlusDecoderImpl::assignExecutionTime(int level, int contextIndex, double executionTimeStack)
{
	double tempStack = 0;
	level -= 1;
	contextData[level].resize(Scheduler::contextCount);

	tempStack += m_aspp->assignExecutionTime(level + 1, contextIndex, 0);

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		contextData[level][i] = ContextData(Scheduler::selectContextByIndex(i));

		contextData[level][i].stackExecutionTime(m_aspp->contextData[level][i]);

		contextData[level][i].stackExecutionTime(o_aspp_seq->contextData[i]);
		contextData[level][i].stackExecutionTime(o_up->contextData[i]);
		contextData[level][i].stackExecutionTime(o_block2->contextData[i]);
	}

	tempStack += o_aspp_seq->getRegulatedExecutionTime(contextIndex);
	tempStack += o_up->getRegulatedExecutionTime(contextIndex);
	tempStack += o_block2->getRegulatedExecutionTime(contextIndex);

	regulatedExecutionTime[level] = tempStack;
	return executionTimeStack + tempStack;
}

double DeepLabV3PlusDecoderImpl::assignDeadline(double quota, int level, int contextIndex, double deadlineStack)
{
	double usedDeadline = 0, tempStack, tempDeadline;

	level -= 1;

	tempStack = m_aspp->assignDeadline(
		(m_aspp->regulatedExecutionTime[level] / regulatedExecutionTime[level]) * quota,
		level + 1, contextIndex, deadlineStack);
	tempDeadline = tempStack - deadlineStack;
	usedDeadline += tempDeadline;
	deadlineStack += tempDeadline;

	o_aspp_seq->_parent->deadlineLogger->info("encoder->aspp-: {:.0f}", tempDeadline);

	o_aspp_seq->relativeDeadline[level] =
		o_aspp_seq->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
	usedDeadline += o_aspp_seq->relativeDeadline[level];
	deadlineStack += o_aspp_seq->relativeDeadline[level];
	o_aspp_seq->stackedDeadline[level] = deadlineStack;

	o_aspp_seq->_parent->deadlineLogger->info("encoder->o_aspp_seq-: {:.0f}", o_aspp_seq->relativeDeadline[level]);

	o_up->relativeDeadline[level] =
		o_up->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
	usedDeadline += o_up->relativeDeadline[level];
	deadlineStack += o_up->relativeDeadline[level];
	o_up->stackedDeadline[level] = deadlineStack;

	o_up->_parent->deadlineLogger->info("encoder->up-: {:.0f}", o_up->relativeDeadline[level]);

	o_block1->relativeDeadline[level] = usedDeadline;
	o_block1->stackedDeadline[level] = deadlineStack;

	o_block1->_parent->deadlineLogger->info("encoder->block1-: {:.0f}", o_block1->relativeDeadline[level]);

	o_block2->relativeDeadline[level] =
		o_block2->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
	usedDeadline += o_block2->relativeDeadline[level];
	deadlineStack += o_block2->relativeDeadline[level];
	o_block2->stackedDeadline[level] = deadlineStack;

	o_block2->_parent->deadlineLogger->info("encoder->block2-: {:.0f}", o_block2->relativeDeadline[level]);

	return deadlineStack;
}

void DeepLabV3PlusDecoderImpl::setAbsoluteDeadline(int level, steady_clock::time_point start, int bias)
{
	m_aspp->setAbsoluteDeadline(level, start, bias);
	o_aspp_seq->setAbsoluteDeadline(level, start, bias);
	o_up->setAbsoluteDeadline(level, start, bias);
	o_block1->setAbsoluteDeadline(level, start, bias);
	o_block2->setAbsoluteDeadline(level, start, bias);
}