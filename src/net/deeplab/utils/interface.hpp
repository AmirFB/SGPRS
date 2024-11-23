# pragma once
# include<torch/torch.h>
# include<torch/script.h>

# include <cnt.hpp>

using namespace FGPRS;

class Backbone : public MyContainer
{
public:
	virtual vector<torch::Tensor> features(torch::Tensor x, int encoder_depth = 5) = 0;
	virtual torch::Tensor features_at(torch::Tensor x, int stage_num) = 0;
	virtual void load_pretrained(string pretrained_path) = 0;
	virtual void make_dilated(vector<int> stage_list, vector<int> dilation_list) = 0;
	virtual ~Backbone() {}
};