#include "pch.h"

#include "GeneratorModel.h"
#include "DiscriminatorModel.h"

// Constants
const int64_t kNoiseSize = 100;						// size of noise vector fed to the generator
const int64_t kBatchSize = 64;						// batch size for training
const int64_t kNumberOfEpochs = 30;					// number of epochs to train
const char* kDataFolder = "./mnist";				// path to MNIST dataset
const int64_t kCheckpointEvery = 200;				// to create new checkpoint periodically
const bool kRestoreFromCheckpoint = false;			// set true to restore from previous checkpoint
const int64_t kNumberOfSamplesPerCheckpoint = 10;	// number of images to sample at every Checkpoint
const int64_t kLogInterval = 10;					// after how many batches to log new update with the loss value


int main()
{
	torch::manual_seed(1);
	
	torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

	DCGANGenerator generator(kNoiseSize);
	generator->to(device);

	DCGANDiscriminator discriminator;
	discriminator->to(device);

	// Loading Data
	auto dataset = torch::data::datasets::MNIST(kDataFolder).map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());
	const int64_t batchesPerEpoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));
	auto dataloader = torch::data::make_data_loader(
		std::move(dataset),
		torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2)	// 2 threads will be used to load the data
	);

	// Training Loop
	torch::optim::Adam generatorOptimizer(generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
	torch::optim::Adam discriminatorOptimizer(discriminator->parameters(), torch::optim::AdamOptions(5e-4).beta1(0.5));

	if (kRestoreFromCheckpoint)
	{
		torch::load(generator, "SavedModels/generator-checkpoint.pt");
		torch::load(generatorOptimizer, "SavedModels/generator-optimizer-checkpoint.pt");
		torch::load(discriminator, "SavedModels/discriminator-checkpoint.pt");
		torch::load(discriminatorOptimizer, "SavedModels/discriminator-optimizer-checkpoint.pt");
	}

	int64_t checkpointCounter = 1;
	for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch)
	{
		int64_t batchIndex = 0;
		for (torch::data::Example<>& batch : *dataloader)
		{
			// Train discriminator with real images
			discriminator->zero_grad();	// to zero out the gradients
			torch::Tensor realImages = batch.data.to(device);
			torch::Tensor realLabels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
			torch::Tensor realOutput = discriminator->forward(realImages);
			torch::Tensor dLossReal = torch::binary_cross_entropy(realOutput, realLabels);
			dLossReal.backward();		// to compute new gradients	// backpropagate

			// Train discriminator with fake images
			torch::Tensor noise = torch::randn({ batch.data.size(0), kNoiseSize,1 , 1 }, device);
			torch::Tensor fakeImages = generator->forward(noise);
			torch::Tensor fakeLabels = torch::zeros(batch.data.size(0), device);
			torch::Tensor fakeOutput = discriminator->forward(fakeImages.detach());
			torch::Tensor dLossFake = torch::binary_cross_entropy(fakeOutput, fakeLabels);
			dLossFake.backward();

			torch::Tensor dLoss = dLossReal + dLossFake;
			discriminatorOptimizer.step();	// progress the discriminator's optimizer by one step to update its parameters

			// Train generator
			generator->zero_grad();
			fakeLabels.fill_(1);	// assign the probabilities close to one	// to fool the discriminator into thinking the images are actually real
			fakeOutput = discriminator->forward(fakeImages);
			torch::Tensor gLoss = torch::binary_cross_entropy(fakeOutput, fakeLabels);
			gLoss.backward();
			generatorOptimizer.step();

			batchIndex++;
			if (batchIndex % kLogInterval == 0)
			{
				std::printf("\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f", epoch, kNumberOfEpochs, batchIndex, batchesPerEpoch, dLoss.item<float>(), gLoss.item<float>());
			}

			if (batchIndex % kCheckpointEvery == 0)
			{
				// Checkpoint the model and optimizer state
				torch::save(generator, "SavedModels/generator-checkpoint.pt");
				torch::save(generatorOptimizer, "SavedModels/generator-optimizer-checkpoint.pt");
				torch::save(discriminator, "SavedModels/discriminator-checkpoint.pt");
				torch::save(discriminatorOptimizer, "SavedModels/discriminator-optimizer-checkpoint.pt");
				
				// Sample the generator and save the images
				torch::Tensor samples = generator->forward(torch::randn({ kNumberOfSamplesPerCheckpoint, kNoiseSize, 1, 1 }, device));
				torch::save((samples + 1.0) / 2.0, torch::str("SavedSamples/dcgan-sample-", checkpointCounter, ".pt"));
				std::cout << "\n-> checkpoint " << checkpointCounter ++<< std::endl;
			}

		}
	}

	std::cout << "\nTraining Complete!" << std::endl;
	std::cin.get();
}





