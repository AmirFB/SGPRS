# include "resnet_encoder.hpp"

BlockImpl::BlockImpl(int64_t inplanes, int64_t planes, int64_t stride_,
	torch::nn::Sequential downsample_, int groups, int base_width, bool _is_basic)
{
	downsample = downsample_;
	stride = stride_;
	int width = int(planes * (base_width / 64.)) * groups;

	conv1 = torch::nn::Conv2d(conv_options(inplanes, width, 3, stride_, 1, groups, false));
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
	conv2 = torch::nn::Conv2d(conv_options(width, width, 3, 1, 1, groups, false));
	bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
	is_basic = _is_basic;
	if (!is_basic)
	{
		conv1 = torch::nn::Conv2d(conv_options(inplanes, width, 1, 1, 0, 1, false));
		conv2 = torch::nn::Conv2d(conv_options(width, width, 3, stride_, 1, groups, false));
		conv3 = torch::nn::Conv2d(conv_options(width, planes * 4, 1, 1, 0, 1, false));
		bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes * 4));
	}

	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("conv2", conv2);
	register_module("bn2", bn2);
	if (!is_basic)
	{
		register_module("conv3", conv3);
		register_module("bn3", bn3);
	}

	if (!downsample->is_empty())
	{
		register_module("downsample", downsample);
	}
}

torch::Tensor BlockImpl::forward(torch::Tensor x)
{
	torch::Tensor residual = x.clone();

	x = conv1->forward(x);
	x = bn1->forward(x);
	x = torch::relu(x);

	x = conv2->forward(x);
	x = bn2->forward(x);

	if (!is_basic)
	{
		x = torch::relu(x);
		x = conv3->forward(x);
		x = bn3->forward(x);
	}

	if (!downsample->is_empty())
	{
		residual = downsample->forward(residual);
	}

	x += residual;
	x = torch::relu(x);

	return x;
}

ResNetEncoderImpl::ResNetEncoderImpl(vector<int> layers, int num_classes, string _model_type, int _groups, int _width_per_group)
{
	model_type = _model_type;
	if (model_type != "resnet18" && model_type != "resnet34")
	{
		expansion = 4;
		is_basic = false;
	}
	if (model_type == "resnext50_32x4d")
	{
		groups = 32; base_width = 4;
	}
	if (model_type == "resnext101_32x8d")
	{
		groups = 32; base_width = 8;
	}
	conv1 = torch::nn::Conv2d(conv_options(3, 64, 7, 2, 3, 1, false));
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
	layer1 = torch::nn::Sequential(_make_layer(64, layers[0]));
	layer2 = torch::nn::Sequential(_make_layer(128, layers[1], 2));
	layer3 = torch::nn::Sequential(_make_layer(256, layers[2], 2));
	layer4 = torch::nn::Sequential(_make_layer(512, layers[3], 2));

	fc = torch::nn::Linear(512 * expansion, num_classes);
	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
	register_module("fc", fc);
}


torch::Tensor  ResNetEncoderImpl::forward(torch::Tensor x)
{
	x = conv1->forward(x);
	x = bn1->forward(x);
	x = torch::relu(x);
	x = torch::max_pool2d(x, 3, 2, 1);

	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);

	x = torch::avg_pool2d(x, 7, 1);
	x = x.view({ x.sizes()[0], -1 });
	x = fc->forward(x);

	return torch::log_softmax(x, 1);
}

vector<torch::nn::Sequential> ResNetEncoderImpl::get_stages()
{
	vector<torch::nn::Sequential> ans;
	ans.push_back(this->layer1);
	ans.push_back(this->layer2);
	ans.push_back(this->layer3);
	ans.push_back(this->layer4);
	return ans;
}

vector<torch::Tensor> ResNetEncoderImpl::features(torch::Tensor x, int encoder_depth)
{
	vector<torch::Tensor> features;
	features.push_back(x);
	x = conv1->forward(x);
	x = bn1->forward(x);
	x = torch::relu(x);
	features.push_back(x);
	x = torch::max_pool2d(x, 3, 2, 1);

	vector<torch::nn::Sequential> stages = get_stages();
	for (int i = 0; i < encoder_depth - 1; i++)
	{
		x = stages[i]->as<torch::nn::Sequential>()->forward(x);
		features.push_back(x);
	}
	//x = layer1->forward(x);
	//features.push_back(x);
	//x = layer2->forward(x);
	//features.push_back(x);
	//x = layer3->forward(x);
	//features.push_back(x);
	//x = layer4->forward(x);
	//features.push_back(x);

	return features;
}

vector<Tensor> ResNetEncoderImpl::forwardSIMO(Tensor input)
{
	return features(input);
}

torch::Tensor ResNetEncoderImpl::features_at(torch::Tensor x, int stage_num)
{
	assert(stage_num > 0 && "the stage number must in range(1,5)");
	x = conv1->forward(x);
	x = bn1->forward(x);
	x = torch::relu(x);
	if (stage_num == 1) return x;
	x = torch::max_pool2d(x, 3, 2, 1);

	x = layer1->forward(x);
	if (stage_num == 2) return x;
	x = layer2->forward(x);
	if (stage_num == 3) return x;
	x = layer3->forward(x);
	if (stage_num == 4) return x;
	x = layer4->forward(x);
	if (stage_num == 5) return x;
	return x;
}

void ResNetEncoderImpl::load_pretrained(string pretrained_path)
{
	map<string, vector<int>> name2layers = getParams();
	ResNetEncoder net_pretrained = ResNetEncoder(name2layers[model_type], 1000, model_type, groups, base_width);
	torch::load(net_pretrained, pretrained_path);

	torch::OrderedDict<string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
	torch::OrderedDict<string, at::Tensor> model_dict = this->named_parameters();

	for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
	{
		if (strstr((*n).key().data(), "fc."))
		{
			continue;
		}
		model_dict[(*n).key()] = (*n).value();
	}

	torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
	auto new_params = model_dict; // implement this
	auto params = this->named_parameters(true /*recurse*/);
	auto buffers = this->named_buffers(true /*recurse*/);
	for (auto& val : new_params)
	{
		auto name = val.key();
		auto* t = params.find(name);
		if (t != nullptr)
		{
			t->copy_(val.value());
		}
		else
		{
			t = buffers.find(name);
			if (t != nullptr)
			{
				t->copy_(val.value());
			}
		}
	}
	torch::autograd::GradMode::set_enabled(true);
	return;
}

torch::nn::Sequential ResNetEncoderImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride)
{

	torch::nn::Sequential downsample;
	if (stride != 1 || inplanes != planes * expansion)
	{
		downsample = torch::nn::Sequential(
			torch::nn::Conv2d(conv_options(inplanes, planes * expansion, 1, stride, 0, 1, false)),
			torch::nn::BatchNorm2d(planes * expansion)
		);
	}
	torch::nn::Sequential layers;
	layers->push_back(Block(inplanes, planes, stride, downsample, groups, base_width, is_basic));
	inplanes = planes * expansion;
	for (int64_t i = 1; i < blocks; i++)
	{
		layers->push_back(Block(inplanes, planes, 1, torch::nn::Sequential(), groups, base_width, is_basic));
	}

	return layers;
}

void ResNetEncoderImpl::make_dilated(vector<int> stage_list, vector<int> dilation_list)
{
	if (stage_list.size() != dilation_list.size())
	{
		cout << "make sure stage list len equal to dilation list len";
		return;
	}
	map<int, torch::nn::Sequential> stage_dict = {};
	stage_dict.insert(pair<int, torch::nn::Sequential>(5, this->layer4));
	stage_dict.insert(pair<int, torch::nn::Sequential>(4, this->layer3));
	stage_dict.insert(pair<int, torch::nn::Sequential>(3, this->layer2));
	stage_dict.insert(pair<int, torch::nn::Sequential>(2, this->layer1));
	for (int i = 0; i < stage_list.size(); i++)
	{
		int dilation_rate = dilation_list[i];
		for (auto m : stage_dict[stage_list[i]]->modules())
		{
			if (m->name() == "torch::nn::Conv2dImpl")
			{
				m->as<torch::nn::Conv2d>()->options.stride(1);
				m->as<torch::nn::Conv2d>()->options.dilation(dilation_rate);
				int kernel_size = m->as<torch::nn::Conv2d>()->options.kernel_size()->at(0);
				m->as<torch::nn::Conv2d>()->options.padding((kernel_size / 2) * dilation_rate);
			}
		}
	}
	return;
}

ResNetEncoder resnet18_encoder(int64_t num_classes)
{
	vector<int> layers = { 2, 2, 2, 2 };
	ResNetEncoder model(layers, num_classes, "resnet18");
	return model;
}

ResNetEncoder resnet34_encoder(int64_t num_classes)
{
	vector<int> layers = { 3, 4, 6, 3 };
	ResNetEncoder model(layers, num_classes, "resnet34");
	return model;
}

ResNetEncoder resnet50_encoder(int64_t num_classes)
{
	vector<int> layers = { 3, 4, 6, 3 };
	ResNetEncoder model(layers, num_classes, "resnet50");
	return model;
}

ResNetEncoder resnet101_encoder(int64_t num_classes)
{
	vector<int> layers = { 3, 4, 23, 3 };
	ResNetEncoder model(layers, num_classes, "resnet101");
	return model;
}

ResNetEncoder pretrained_resnet(int64_t num_classes, string model_name, string weight_path)
{
	map<string, vector<int>> name2layers = getParams();
	int groups = 1;
	int width_per_group = 64;
	if (model_name == "resnext50_32x4d")
	{
		groups = 32; width_per_group = 4;
	}
	if (model_name == "resnext101_32x8d")
	{
		groups = 32; width_per_group = 8;
	}
	ResNetEncoder net_pretrained = ResNetEncoder(name2layers[model_name], 1000, model_name, groups, width_per_group);
	torch::load(net_pretrained, weight_path);
	if (num_classes == 1000) return net_pretrained;
	ResNetEncoder module = ResNetEncoder(name2layers[model_name], num_classes, model_name);

	torch::OrderedDict<string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
	torch::OrderedDict<string, at::Tensor> model_dict = module->named_parameters();

	for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
	{
		if (strstr((*n).key().data(), "fc."))
		{
			continue;
		}
		model_dict[(*n).key()] = (*n).value();
	}

	torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
	auto new_params = model_dict; // implement this
	auto params = module->named_parameters(true /*recurse*/);
	auto buffers = module->named_buffers(true /*recurse*/);
	for (auto& val : new_params)
	{
		auto name = val.key();
		auto* t = params.find(name);
		if (t != nullptr)
		{
			t->copy_(val.value());
		}
		else
		{
			t = buffers.find(name);
			if (t != nullptr)
			{
				t->copy_(val.value());
			}
		}
	}
	torch::autograd::GradMode::set_enabled(true);
	return module;
}