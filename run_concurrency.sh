bash build.sh

echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,garbage_collection_threshold:0.99
sudo nvidia-smi -pl 280

# ./build/fgprs clear concurrency

./build/fgprs concurrency 1 1 1000
./build/fgprs concurrency 1 1 1000
# ./build/fgprs concurrency 1 5 1000
# ./build/fgprs concurrency 2 5 1000
# ./build/fgprs concurrency 3 5 1000
# ./build/fgprs concurrency 4 5 1000
# ./build/fgprs concurrency 1 1 500
# ./build/fgprs concurrency 1 5 100

./build/fgprs clear concurrency

clear

for mode in {1..3}
do
	for count in {1..5}
	do
		./build/fgprs concurrency $mode $count 500
	done
done

echo quit | nvidia-cuda-mps-control