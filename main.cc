#include <torch/torch.h>
#include <iostream>

# include "nvToolsExt.h"

# include <torch/torch.h>
# include <torch/script.h>
# include <c10/cuda/CUDAStream.h>
# include <ATen/cuda/CUDAContext.h>
# include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/Event.h>
#include <c10/core/impl/InlineEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <c10/util/irange.h>

#include <cuda_runtime.h>

# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>
#include <cuda_profiler_api.h>

# include <torch/torch.h>
# include <c10/cuda/CUDAStream.h>
# include <cuda_runtime_api.h>
# include <nvToolsExt.h>

#include <torch/torch.h>
#include <iostream>
#include <future>

void multiplyMatrix
(torch::Tensor matrix1, torch::Tensor matrix2, torch::Tensor result,
	c10::cuda::CUDAStream stream, const char* name, bool highPriority = false)
{
	stream = c10::cuda::getStreamFromPool(highPriority);
	at::cuda::CUDAStreamGuard guard(stream);
	// at::cuda::setCurrentCUDAStream(stream);

	for (int i = 0; i < 1000; i++)
	{
		// stream = c10::cuda::getStreamFromPool();
#ifdef ENABLE_NVTX_PROFILING
		auto id = nvtxRangeStartA(name);
#endif

		result = torch::mm(matrix1, matrix2);
		stream.synchronize();

#ifdef ENABLE_NVTX_PROFILING
		nvtxRangeEnd(id);
#endif
	}
}

int main()
{
	// Initialize CUDA device and create two streams
	torch::Device device(torch::kCUDA);
	c10::cuda::CUDAStream  stream1 = c10::cuda::getStreamFromPool();
	c10::cuda::CUDAStream  stream2 = c10::cuda::getStreamFromPool();
	c10::cuda::CUDAStream  stream3 = c10::cuda::getStreamFromPool();
	c10::cuda::CUDAStream  stream4 = c10::cuda::getStreamFromPool(true, 0);
	c10::cuda::CUDAStream  stream5 = c10::cuda::getStreamFromPool(true, 0);

	// Matrix dimensions
	int N = 1000;
	int M = 1000;
	int K = 1000;

	// Create tensors on the CUDA device
	torch::TensorOptions options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
	torch::Tensor matrix1 = torch::randn({ N, M }, options);
	torch::Tensor matrix2 = torch::randn({ M, K }, options);
	torch::Tensor result1 = torch::zeros({ N, K }, options);
	torch::Tensor result2 = torch::zeros({ N, K }, options);
	torch::Tensor result3 = torch::zeros({ N, K }, options);
	torch::Tensor result4 = torch::zeros({ N, K }, options);
	torch::Tensor result5 = torch::zeros({ N, K }, options);

	// Launch matrix multiplications asynchronously
	std::future<void> future1 = std::async(std::launch::async, multiplyMatrix, matrix1, matrix2, result1, stream1, "111", false);
	std::future<void> future2 = std::async(std::launch::async, multiplyMatrix, matrix1, matrix2, result2, stream2, "222", false);
	std::future<void> future3 = std::async(std::launch::async, multiplyMatrix, matrix1, matrix2, result2, stream5, "333", false);
	std::future<void> future4 = std::async(std::launch::async, multiplyMatrix, matrix1, matrix2, result2, stream2, "444", false);
	std::future<void> future5 = std::async(std::launch::async, multiplyMatrix, matrix1, matrix2, result2, stream2, "555", true);

	// Wait for the futures to complete
	future1.wait();
	future2.wait();
	future3.wait();
	future4.wait();
	future5.wait();

	cudaProfilerStart();
	// Launch matrix multiplications asynchronously
	std::future<void> future11 = std::async(std::launch::async, multiplyMatrix, matrix1, matrix2, result1, stream1, "111", false);
	std::future<void> future12 = std::async(std::launch::async, multiplyMatrix, matrix1, matrix2, result2, stream2, "222", false);
	std::future<void> future13 = std::async(std::launch::async, multiplyMatrix, matrix1, matrix2, result2, stream5, "333", false);
	std::future<void> future14 = std::async(std::launch::async, multiplyMatrix, matrix1, matrix2, result2, stream2, "444", false);
	std::future<void> future15 = std::async(std::launch::async, multiplyMatrix, matrix1, matrix2, result2, stream2, "555", true);

	// Wait for the futures to complete
	future11.wait();
	future12.wait();
	future13.wait();
	future14.wait();
	future15.wait();
	cudaProfilerStop();

	// // Print a few elements from the results
	// std::cout << "Result 1:" << std::endl;
	// std::cout << result1.slice(/*dim=*/0, /*start=*/0, /*end=*/4).slice(/*dim=*/1, /*start=*/0, /*end=*/4) << std::endl;

	// std::cout << "Result 2:" << std::endl;
	// std::cout << result2.slice(/*dim=*/0, /*start=*/0, /*end=*/4).slice(/*dim=*/1, /*start=*/0, /*end=*/4) << std::endl;

	return 0;
}
