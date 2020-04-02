# DCGANetwork
Deep Convolution Generative Adversarial Network implemented in C++ using Pytorch C++ API

# Setting up PyTorch C++ API for VS
1. Download pre-built libtorch from ‘ https://pytorch.org/ ’ (gpu[cuda] or cpu[none])
2. Extract the downloaded zip file.
3. In Visual Studio project properties, Configuration must be either release or debug. Platform must be x64.
4. In VC++ Directories and ‘C/C++ →Additional Include Directories’, add \path\to\libtorch\include and \path\to\libtorch\csrc\api\include .
5. In Linker → Additional Library Directories, add \path\to\libtorch\lib .
6. In Linker → Input, add torch.lib; caffe2_module_test_dynamic.lib; c10.lib
7. Run the project in x64 platform
8. You will also have to copy .dll files from \path\to\libtorch\lib to bin\Debug-x64\DCGANetwork
