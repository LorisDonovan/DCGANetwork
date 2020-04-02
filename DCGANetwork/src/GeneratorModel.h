#pragma once

class DCGANGeneratorImpl : public torch::nn::Module
{
public:
	DCGANGeneratorImpl(int kNoiseSize)
		:m_Conv1(torch::nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)),
		m_BatchNorm1(256),
		m_Conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
		m_BatchNorm2(128),
		m_Conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
		m_BatchNorm3(64),
		m_Conv4(torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false))
	{
		// register_module() is needed if we want to use the parameter() method later on
		register_module("m_Conv1", m_Conv1);
		register_module("m_Conv2", m_Conv2);
		register_module("m_Conv3", m_Conv3);
		register_module("m_Conv4", m_Conv4);

		register_module("m_BatchNorm1", m_BatchNorm1);
		register_module("m_BatchNorm2", m_BatchNorm2);
		register_module("m_BatchNorm3", m_BatchNorm3);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(m_BatchNorm1(m_Conv1(x)));
		x = torch::relu(m_BatchNorm2(m_Conv2(x)));
		x = torch::relu(m_BatchNorm3(m_Conv3(x)));
		x = torch::tanh(m_Conv4(x));

		return x;
	}

private:
	torch::nn::ConvTranspose2d m_Conv1;
	torch::nn::ConvTranspose2d m_Conv2;
	torch::nn::ConvTranspose2d m_Conv3;
	torch::nn::ConvTranspose2d m_Conv4;

	torch::nn::BatchNorm2d m_BatchNorm1;
	torch::nn::BatchNorm2d m_BatchNorm2;
	torch::nn::BatchNorm2d m_BatchNorm3;
};
TORCH_MODULE(DCGANGenerator);	// Defining the class DCGANGeneratorImpl
