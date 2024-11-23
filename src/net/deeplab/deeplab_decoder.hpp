# pragma once
# include "util.hpp"

# include <sqnl.hpp>

using namespace torch;
using namespace nn;
using namespace FGPRS;

MySequential ASPPConv(int in_channels, int out_channels, int dilation);

MySequential SeparableConv2d(int in_channels, int out_channels, int kernel_size, int stride = 1,
	int padding = 0, int dilation = 1, bool bias = true);

MySequential ASPPSeparableConv(int in_channels, int out_channels, int dilation);

class ASPPPoolingImpl : public MyContainer
{
public:
	MySequential seq{ nullptr };
	ASPPPoolingImpl(int in_channels, int out_channels);
	torch::Tensor forward(torch::Tensor x);

}; TORCH_MODULE(ASPPPooling);

class ASPPImpl : public MyContainer
{
public:
	ASPPImpl(int in_channels, int out_channels, vector<int> atrous_rates, bool separable = false);
	torch::Tensor forward(torch::Tensor x);
private:
	vector<MySequential> m_modules;
	ASPPPooling m_aspppooling{ nullptr };
	MySequential m_project{ nullptr };

	vector<shared_ptr<Operation>> o_modules;
	shared_ptr<Operation> o_aspppooling, o_project, temp;
public:
	void assignOperations(MyContainer* owner) override;
	Tensor schedule(Tensor input, int leve) override;
	Tensor analyze(int warmup, int repeat, Tensor input, int index, int level) override;
	double assignExecutionTime(int level, int contextIndex, double executionTimeStack) override;
	double assignDeadline(double quota, int level, int contextIndex, double deadlineStack) override;
	void setAbsoluteDeadline(int level, steady_clock::time_point start, int bias) override;
}; TORCH_MODULE(ASPP);

class DeepLabV3DecoderImpl : public MyContainer
{
public:
	DeepLabV3DecoderImpl(int in_channels, int out_channels = 256, vector<int> atrous_rates = { 12, 24, 36 });
	torch::Tensor forward(vector<torch::Tensor> x);
	int out_channels = 0;
private:
	MySequential seq{};
}; TORCH_MODULE(DeepLabV3Decoder);

class DeepLabV3PlusDecoderImpl : public MyContainer
{
public:
	DeepLabV3PlusDecoderImpl(vector<int> encoder_channels, int out_channels,
		vector<int> atrous_rates, int output_stride = 16);
	torch::Tensor forward(vector< torch::Tensor> x);
private:
	ASPP m_aspp{ nullptr };
	MySequential m_aspp_seq{ nullptr };
	Upsample m_up{ nullptr };
	MySequential m_block1{ nullptr };
	MySequential m_block2{ nullptr };

	shared_ptr<Operation> o_aspp, o_aspp_seq, o_up, o_block1, o_block2;

public:
	void assignOperations(MyContainer* owner) override;
	Tensor scheduleMISO(vector<Tensor> inputs, int leve);
	Tensor analyzeMISO(int warmup, int repeat, vector<Tensor> inputs, int index, int level);
	double assignExecutionTime(int level, int contextIndex, double executionTimeStack) override;
	double assignDeadline(double quota, int level, int contextIndex, double deadlineStack) override;
	void setAbsoluteDeadline(int level, steady_clock::time_point start, int bias) override;
}; TORCH_MODULE(DeepLabV3PlusDecoder);