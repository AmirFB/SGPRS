bash build.sh

echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export CUDA_LAUNCH_BLOCKING=1
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

./build/fgprs clear interference

for var in {2..68..2}
do
	./build/fgprs interference $var 100
done

echo quit | nvidia-cuda-mps-control