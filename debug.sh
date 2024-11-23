shell echo quit | nvidia-cuda-mps-control
shell nvidia-cuda-mps-control -d
shell export CUDA_LAUNCH_BLOCKING=1
shell export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
shell export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1

shell sudo nvidia-smi -pl 280

shell ./build/fgprs proposed
	
shell echo quit | nvidia-cuda-mps-control