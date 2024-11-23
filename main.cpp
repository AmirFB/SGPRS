#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

// Define Basic Block
struct SimpleBlock : torch::nn::Module
{
	torch::nn::Conv2d _conv1{ nullptr }, _conv2{ nullptr };
	torch::nn::BatchNorm2d _bn1{ nullptr }, _bn2{ nullptr };
	torch::nn::Sequential _downsample{ nullptr };

	SimpleBlock(int64_t in_channels, int64_t out_channels, int64_t stride = 1, torch::nn::Sequential downsample = torch::nn::Sequential{ nullptr })
	{
		_conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1).bias(false)));
		_bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
		_conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1).bias(false)));
		_bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));

		if (!downsample.is_empty())
			_downsample = register_module("downsample", downsample);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		torch::Tensor residual = x;
		// printf("Input dimenstions: %d, %d\n", x.size(1), x.size(2));
		x = torch::relu(_bn1(_conv1(x)));
		x = _bn2(_conv2(x));

		if (!_downsample.is_empty())
			residual = _downsample->forward(residual);

		x += residual;
		x = torch::relu(x);
		// printf("Output dimenstions: %d, %d\n", x.size(1), x.size(2));
		return x;
	}
};

// Define Bottleneck Block
struct BottleneckBlock : torch::nn::Module
{
	torch::nn::Conv2d _conv1{ nullptr }, _conv2{ nullptr }, _conv3{ nullptr };
	torch::nn::BatchNorm2d _bn1{ nullptr }, _bn2{ nullptr }, _bn3{ nullptr };
	torch::nn::Sequential _downsample{ nullptr };

	BottleneckBlock(int64_t in_channels, int64_t out_channels, int64_t stride = 1, torch::nn::Sequential downsample = torch::nn::Sequential())
	{
		_conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels / 4, 1).bias(false)));
		_bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels / 4));
		_conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels / 4, out_channels / 4, 3).stride(stride).padding(1).bias(false)));
		_bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels / 4));
		_conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels / 4, out_channels, 1).bias(false)));
		_bn3 = register_module("bn3", torch::nn::BatchNorm2d(out_channels));

		if (!downsample.is_empty())
			_downsample = register_module("downsample", downsample);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		torch::Tensor residual = x;

		x = torch::relu(_bn1(_conv1(x)));
		x = torch::relu(_bn2(_conv2(x)));
		x = _bn3(_conv3(x));

		if (!_downsample.is_empty())
			residual = _downsample->forward(residual);

		x += residual;
		x = torch::relu(x);

		return x;
	}
};

// Define FCSoftmaxModule
struct FCSoftmaxModule : torch::nn::Module
{
	torch::nn::Linear fc{ nullptr };

	FCSoftmaxModule(int64_t in_features, int64_t num_classes)
	{
		fc = register_module("fc", torch::nn::Linear(in_features, num_classes));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::adaptive_avg_pool2d(x, { 1, 1 });
		x = x.view({ x.size(0), -1 });
		x = fc(x);
		// return torch::log_softmax(x, /*dim=*/1);
		return x;
	}
};

// Define ResNet Model
struct ResNet : torch::nn::Module
{
	int64_t in_channels{ 3 };  // For RGB images
	int64_t num_classes{ 1000 };
	torch::nn::Sequential _layer1{ nullptr };
	torch::nn::Sequential _layer2{ nullptr };
	torch::nn::Sequential _layer3{ nullptr };
	torch::nn::Sequential _layer4{ nullptr };

	ResNet(const std::vector<int>& layer_sizes, int block_type, const std::vector<int>& num_blocks)
	{
		// Create layer1 using make_layer
		auto layer1 = torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3).bias(false)),
			torch::nn::BatchNorm2d(64),
			torch::nn::ReLU(),
			torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1))
		);

		layer1 = make_layer(layer1, layer_sizes[0], block_type, num_blocks[0]);
		_layer1 = register_module("layer1", layer1);

		// // Layer2
		auto layer2 = torch::nn::Sequential();
		layer2 = make_layer(layer2, layer_sizes[1], block_type, num_blocks[1], 2);
		_layer2 = register_module("layer2", layer2);

		/// Layer3
		auto layer3 = torch::nn::Sequential();
		layer3 = make_layer(layer3, layer_sizes[2], block_type, num_blocks[2], 2);
		_layer3 = register_module("layer3", layer3);

		// Create layer4 using make_layer
		auto layer4 = torch::nn::Sequential();
		layer4 = make_layer(layer4, layer_sizes[3], block_type, num_blocks[3], 2);
		layer4->push_back(FCSoftmaxModule(layer_sizes[3], num_classes));
		_layer4 = register_module("layer4", layer4);
	}

	torch::nn::Sequential make_layer(torch::nn::Sequential& layer, int64_t channels, int block_type, int num_blocks, int64_t stride = 1)
	{
		auto downsample = torch::nn::Sequential{ nullptr };
		int64_t in_channels = channels == 64 ? 64 : channels / 2;

		if (stride != 1 || in_channels != channels)
			downsample = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, channels, 1).stride(stride).bias(false)),
				torch::nn::BatchNorm2d(channels)
			);

		layer->push_back(SimpleBlock(in_channels, channels, stride, downsample));

		in_channels = channels;
		stride = 1;  // Only the first block has stride > 1

		for (int i = 0; i < (num_blocks - 1); ++i)
		{
			if (block_type == 18 || block_type == 34)
				layer->push_back(SimpleBlock(in_channels, channels, stride));

			else
				layer->push_back(BottleneckBlock(in_channels, channels, stride));
		}

		return layer;
	}

	torch::Tensor forward(torch::Tensor& x)
	{
		x = _layer1->forward(x);//printf("layer 1 done\n\n");
		x = _layer2->forward(x);//printf("layer 2 done\n\n");
		x = _layer3->forward(x);//printf("layer 3 done\n\n");
		return _layer4->forward(x);//printf("layer 4 done\n\n");

		// return x;
	}
};

// Factory function to create ResNet instances
ResNet create_resnet(int resnet_type)
{
	switch (resnet_type)
	{
		case 18:
			return ResNet({ 64, 128, 256, 512 }, 18, { 2, 2, 2, 2 });
		case 34:
			return ResNet({ 64, 128, 256, 512 }, 34, { 3, 5, 6, 3 });
		case 50:
			return ResNet({ 256, 512, 1024, 2048 }, 50, { 3, 4, 6, 3 });
		case 101:
			return ResNet({ 256, 512, 1024, 2048 }, 101, { 3, 5, 23, 3 });
		case 152:
			return ResNet({ 256, 512, 1024, 2048 }, 152, { 3, 8, 36, 3 });
		default:
			throw std::runtime_error("Unsupported ResNet type");
	}
}

// Benchmark function
void benchmark_resnet(int64_t input_size, int64_t batch_size, int64_t num_iterations, int resnet_type)
{
	torch::NoGradGuard no_grad;
	// Check if CUDA is available
	if (!torch::cuda::is_available())
	{
		std::cerr << "CUDA is not available. Exiting." << std::endl;
		return;
	}

	// Set the device to CUDA
	torch::Device device(torch::kCUDA);

	// Create a ResNet model
	ResNet model = create_resnet(resnet_type);
	model.to(device);  // Move the model to CUDA

	// Dummy input tensor
	torch::Tensor input = torch::randn({ batch_size, 3, input_size, input_size });
	input = input.to(device);  // Move the input tensor to CUDA

	// Warm-up
	for (int i = 0; i < 5; ++i)
	{
		torch::NoGradGuard no_grad;
		model.forward(input);
	}

	// Benchmark
	torch::Tensor dummy;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < num_iterations; ++i)
	{
		torch::NoGradGuard no_grad;
		dummy = model.forward(input);
	}
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	double fps = static_cast<double>(num_iterations) / duration.count() * 1000;

	std::cout << "ResNet-" << resnet_type << " - Input Size: " << input_size << "x" << input_size
		<< ", Batch Size: " << batch_size << ", FPS: " << fps << std::endl;
}

int main()
{
	printf("Let's start\n");
	// ResNet variations to benchmark
	std::vector<int> resnet_types = { 18, 34 };//, 50, 101, 152 };

	// Common benchmark parameters
	int64_t input_size = 224;
	int64_t batch_size = 1;
	int64_t num_iterations = 100;

	for (int resnet_type : resnet_types)
	{
		printf("resnet_type: %d\n", resnet_type);
		benchmark_resnet(input_size, batch_size, num_iterations, resnet_type);
	}

	return 0;
}