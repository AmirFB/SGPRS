# pragma once
# include "resnet_encoder.hpp"
# include "vgg_encoder.hpp"
# include "deeplab_decoder.hpp"

# include <cnt.hpp>

using namespace std;

using namespace FGPRS;

class DeepLabV3Impl : public MyContainer
{
public:
	DeepLabV3Impl() {}
	~DeepLabV3Impl()
	{
		//delete encoder;
	}
	DeepLabV3Impl(int num_classes, string encoder_name = "resnet18", /*string pretrained_path = "", */ int encoder_depth = 5,
		int decoder_channels = 256, int in_channels = 3, double upsampling = 8);
	torch::Tensor forward(torch::Tensor x);
private:
	Backbone* encoder;
	DeepLabV3Decoder decoder{ nullptr };
	SegmentationHead head{ nullptr };
	int num_classes = 1;
}; TORCH_MODULE(DeepLabV3);

class DeepLabV3PlusImpl : public MyContainer
{
public:
	DeepLabV3PlusImpl() {}
	~DeepLabV3PlusImpl()
	{
		//delete encoder;
	}
	DeepLabV3PlusImpl(int num_classes, string encoder_name = "resnet18", /*string pretrained_path = "", */ int encoder_depth = 5,
		int encoder_output_stride = 16, int decoder_channels = 256, int in_channels = 3, double upsampling = 4);
	torch::Tensor forward(torch::Tensor x);
private:
	Backbone* encoder;
	shared_ptr<Backbone> m_encoder;
	DeepLabV3PlusDecoder m_decoder{ nullptr };
	SegmentationHead m_head{ nullptr };
	shared_ptr<Operation> o_encoder, o_decoder, o_head;

	int num_classes = 1;
	vector<int> decoder_atrous_rates = { 12, 24, 36 };

public:
	void assignOperations() override;
	Tensor schedule(Tensor input, int level) override;
	Tensor analyze(int warmup, int repeat, Tensor input, int index, int level) override;
	double assignExecutionTime(int level, int contextIndex, double executionTimeStack) override;
	double assignDeadline(double quota, int level, int contextIndex, double deadlineStack) override;
	void setAbsoluteDeadline(int level, steady_clock::time_point start, int bias) override;
}; TORCH_MODULE(DeepLabV3Plus);