bash build.sh

echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
# export CUDA_LAUNCH_BLOCKING=0
export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# export PYTORCH_NO_CUDA_MEMORY_CACHING=0

sudo nvidia-smi -pl 280

./build/fgprs speedup	2 2500
./build/fgprs speedup	68 2500
./build/fgprs clear speedup

for var in {2..68..2}
do
	./build/fgprs speedup	$var 2500
done

echo quit | nvidia-cuda-mps-control