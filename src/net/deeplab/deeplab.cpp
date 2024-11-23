# include "main.hpp"
# include "deeplab.hpp"

# include <cnt.hpp>
# include <schd.hpp>

using namespace std;
using namespace FGPRS;

DeepLabV3Impl::DeepLabV3Impl(int _num_classes, string encoder_name, /*string pretrained_path,*/ int encoder_depth,
	int decoder_channels, int in_channels, double upsampling)
{
	num_classes = _num_classes;
	auto encoder_param = encoder_params();
	vector<int> encoder_channels = encoder_param[encoder_name]["out_channels"];
	if (!encoder_param.contains(encoder_name))
		cout << "encoder name must in {resnet18, resnet34, resnet50, resnet101, resnet150, \
				resnext50_32x4d, resnext101_32x8d, vgg11, vgg11_bn, vgg13, vgg13_bn, \
				vgg16, vgg16_bn, vgg19, vgg19_bn,}";
	if (encoder_param[encoder_name]["class_type"] == "resnet")
		encoder = new ResNetEncoderImpl(encoder_param[encoder_name]["layers"], 1000, encoder_name);
	else if (encoder_param[encoder_name]["class_type"] == "vgg")
		encoder = new VGGImpl(encoder_param[encoder_name]["cfg"], 1000, encoder_param[encoder_name]["batch_norm"]);
	else cout << "unknown error in backbone initialization";

	// encoder->load_pretrained(pretrained_path);
	encoder->make_dilated({ 5,4 }, { 4,2 });

	decoder = DeepLabV3Decoder(encoder_channels[encoder_channels.size() - 1], decoder_channels);
	head = SegmentationHead(decoder_channels, num_classes, 1, upsampling);

	register_module("encoder", shared_ptr<Backbone>(encoder));
	register_module("decoder", decoder);
	register_module("head", head);
}

torch::Tensor DeepLabV3Impl::forward(torch::Tensor x)
{
	vector<torch::Tensor> features = encoder->features(x);
	x = decoder->forward(features);
	x = head->forward(x);
	return x;
}

DeepLabV3PlusImpl::DeepLabV3PlusImpl(int _num_classes, string encoder_name, /*string pretrained_path,*/ int encoder_depth,
	int encoder_output_stride, int decoder_channels, int in_channels, double upsampling)
{
	num_classes = _num_classes;
	auto encoder_param = encoder_params();
	vector<int> encoder_channels = encoder_param[encoder_name]["out_channels"];
	if (!encoder_param.contains(encoder_name))
		cout << "encoder name must in {resnet18, resnet34, resnet50, resnet101, resnet150, \
				resnext50_32x4d, resnext101_32x8d, vgg11, vgg11_bn, vgg13, vgg13_bn, \
				vgg16, vgg16_bn, vgg19, vgg19_bn,}";
	if (encoder_param[encoder_name]["class_type"] == "resnet")
		encoder = new ResNetEncoderImpl(encoder_param[encoder_name]["layers"], 1000, encoder_name);
	else if (encoder_param[encoder_name]["class_type"] == "vgg")
		encoder = new VGGImpl(encoder_param[encoder_name]["cfg"], 1000, encoder_param[encoder_name]["batch_norm"]);
	else cout << "unknown error in backbone initialization";

	// encoder->load_pretrained(pretrained_path);
	if (encoder_output_stride == 8)
	{
		encoder->make_dilated({ 5,4 }, { 4,2 });
	}
	else if (encoder_output_stride == 16)
	{
		encoder->make_dilated({ 5 }, { 2 });
	}
	else
	{
		cout << "Encoder output stride should be 8 or 16";
	}

	// m_decoder = DeepLabV3PlusDecoder(encoder_channels, decoder_channels, decoder_atrous_rates, encoder_output_stride);
	// m_head = SegmentationHead(decoder_channels, num_classes, 1, upsampling);

	// m_encoder = *register_module("encoder", shared_ptr<Backbone>(encoder));
	// register_module("encoder", shared_ptr<Backbone>(encoder));
	m_encoder = register_module("encoder", shared_ptr<Backbone>(encoder));
	m_decoder = register_module("decoder", DeepLabV3PlusDecoder(encoder_channels, decoder_channels, decoder_atrous_rates, encoder_output_stride));
	m_head = register_module("head", SegmentationHead(decoder_channels, num_classes, 1, upsampling));
}

torch::Tensor DeepLabV3PlusImpl::forward(torch::Tensor x)
{
	vector<torch::Tensor> features = m_encoder->features(x);
	x = m_decoder->forward(features);
	x = m_head->forward(x);
	return x;
}

void DeepLabV3PlusImpl::assignOperations()
{
	o_encoder = addOperationSIMO(this, "encoder", m_encoder, 0, false, false);

	m_decoder->assignOperations(this);
	// copyOperations("decoder", (MyContainer*)&m_decoder);
	// o_decoder = addOperationSIMO(this, "decoder", m_decoder.ptr(), 3);

	o_head = addOperation(this, "head", m_head.ptr(), 0, true);
}

Tensor DeepLabV3PlusImpl::schedule(Tensor input, int level)
{
	auto features = o_encoder->scheduleSIMOSync(input);

	// if (level == 3)
	// 	input = o_decoder->scheduleSync(features);

	// else
	input = m_decoder->scheduleMISO(features, level);
	return o_head->scheduleSync(input);
}

Tensor DeepLabV3PlusImpl::analyze(int warmup, int repeat, Tensor input, int index, int level)
{
	auto features = o_encoder->analyzeSIMO(warmup, repeat, input, index);
	// if (level == 3)
	// 	input = o_decoder->analyze(warmup, repeat, input, index);

	// else
	input = m_decoder->analyzeMISO(warmup, repeat, features, index, level);
	input = o_head->analyze(warmup, repeat, input, index);

	return input;
}

double DeepLabV3PlusImpl::assignExecutionTime(int level, int contextIndex, double executionTimeStack)
{
	double tempStack = 0;

	level -= 1;

	contextData[level].resize(Scheduler::contextCount);

	for (int i = 0; i < Scheduler::contextCount; i++)
	{
		contextData[level][i] = ContextData(Scheduler::selectContextByIndex(i));
		contextData[level][i].stackExecutionTime(o_encoder->contextData[i]);
	}

	tempStack += o_encoder->getRegulatedExecutionTime(contextIndex);

	tempStack += m_decoder->assignExecutionTime(level + 1, contextIndex, 0);

	for (int i = 0; i < Scheduler::contextCount; i++)
		contextData[level][i].stackExecutionTime(o_head->contextData[i]);

	tempStack += o_head->getRegulatedExecutionTime(contextIndex);

	regulatedExecutionTime[level] = tempStack;
	return executionTimeStack + tempStack;
}

double DeepLabV3PlusImpl::assignDeadline(double quota, int level, int contextIndex, double deadlineStack)
{
	double usedDeadline = 0, tempStack, tempDeadline;

	level -= 1;
	deadlineLogger->info("encoder reg-: {:.0f}", o_encoder->getRegulatedExecutionTime(contextIndex));
	o_encoder->relativeDeadline[level] =
		o_encoder->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
	usedDeadline += o_encoder->relativeDeadline[level];
	deadlineStack += o_encoder->relativeDeadline[level];
	o_encoder->stackedDeadline[level] = deadlineStack;

	deadlineLogger->info("encoder-: {:.0f}", o_encoder->relativeDeadline[level]);

	tempStack = m_decoder->assignDeadline(
		(m_decoder->regulatedExecutionTime[level] / regulatedExecutionTime[level]) * quota,
		level + 1, contextIndex, deadlineStack);
	tempDeadline = tempStack - deadlineStack;
	usedDeadline += tempDeadline;
	deadlineStack += tempDeadline;

	deadlineLogger->info("decoder-: {:.0f}", tempDeadline);

	o_head->relativeDeadline[level] =
		o_head->getRegulatedExecutionTime(contextIndex) / regulatedExecutionTime[level] * quota;
	usedDeadline += o_head->relativeDeadline[level];
	deadlineStack += o_head->relativeDeadline[level];
	o_head->stackedDeadline[level] = deadlineStack;

	deadlineLogger->info("head-: {:.0f}", o_head->relativeDeadline[level]);

	return deadlineStack;
}

void DeepLabV3PlusImpl::setAbsoluteDeadline(int level, steady_clock::time_point start, int bias)
{
	o_encoder->setAbsoluteDeadline(level, start, bias);
	m_decoder->setAbsoluteDeadline(level, start, bias);
	o_head->setAbsoluteDeadline(level, start, bias);
}