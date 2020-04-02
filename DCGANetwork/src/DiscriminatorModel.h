#pragma once

class DCGANDiscriminatorImpl : public torch::nn::Module
{
public:
	DCGANDiscriminatorImpl()
		:m_Conv1(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
		m_Conv2(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
		m_Conv3(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
		m_Conv4(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false))
	{
		// register_module() is needed if we want to use the parameter() method later on
		register_module("m_Conv1", m_Conv1);
		register_module("m_Conv2", m_Conv2);
		register_module("m_Conv3", m_Conv3);
		register_module("m_Conv4", m_Conv4);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::leaky_relu(m_Conv1(x), 0.2);
		x = torch::leaky_relu(m_Conv2(x), 0.2);
		x = torch::leaky_relu(m_Conv3(x), 0.2);
		x = torch::sigmoid(m_Conv4(x));

		return x;
	}

private:
	torch::nn::Conv2d m_Conv1;
	torch::nn::Conv2d m_Conv2;
	torch::nn::Conv2d m_Conv3;
	torch::nn::Conv2d m_Conv4;
};
TORCH_MODULE(DCGANDiscriminator);	// Defining the class DCGANDiscriminatorImpl

// OR // put this in main:

// Descriminator Module
//torch::nn::Sequential discriminator(
//	// Layer 1
//	torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
//	torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
//	// Layer 2
//	torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
//	torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
//	// Layer 3
//	torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
//	torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
//	// Layer 4
//	torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
//	torch::nn::Sigmoid()
//);
