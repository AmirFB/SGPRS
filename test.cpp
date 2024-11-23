#include <iostream>

# include <cuda_runtime_api.h>
# include <cuda.h>
# include <cudaTypedefs.h>
# include <cuda_runtime.h>
# include <driver_types.h>

using namespace std;

int main()
{
	CUcontext cu_ctx, cu_ctx2, cu_ctx3;
	CUexecAffinityParam_v1 affinity;
	affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
	affinity.param.smCount.val = 40;
	cuInit(0);
	auto result = cuCtxCreate_v3(&cu_ctx, &affinity, 1, 0, 0);
	cuInit(0);

	cout << "Result: " << result << endl;

	// Retrieve the extra data from the CUcontext.extra field and print it
	// int* retrieved_extra_data = (int*)cu_ctx->extra;
	// std::cout << "Extra data value: " << *retrieved_extra_data << std::endl;
	size_t dummy;

	result = cuCtxGetLimit(&dummy, CU_LIMIT_PRINTF_FIFO_SIZE);
	std::cout << "Index: " << ((dummy % 0X1000) >> 8) << "\tResult: " << result << std::endl;

	result = cuCtxGetLimit(&dummy, CU_LIMIT_PRINTF_FIFO_SIZE);
	cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, dummy + 256 * 7);
	result = cuCtxGetLimit(&dummy, CU_LIMIT_PRINTF_FIFO_SIZE);
	std::cout << "Index: " << ((dummy % 0X1000) >> 8) << "\tResult: " << result << std::endl;


	result = cuCtxCreate_v3(&cu_ctx2, &affinity, 1, 0, 0);
	cuInit(0);

	result = cuCtxGetLimit(&dummy, CU_LIMIT_PRINTF_FIFO_SIZE);
	cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, dummy + 256 * 5);
	result = cuCtxGetLimit(&dummy, CU_LIMIT_PRINTF_FIFO_SIZE);
	std::cout << "Index: " << ((dummy % 0X1000) >> 8) << "\tResult: " << result << std::endl;

	// Free the extra data memory
	// delete retrieved_extra_data;

	return 0;
}
