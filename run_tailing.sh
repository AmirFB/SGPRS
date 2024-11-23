bash build.sh

echo quit | nvidia-cuda-mps-control
nvidia-cuda-mps-control -d
export CUDA_LAUNCH_BLOCKING=1
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

./build/fgprs clear tailing

for var in {2..68..2}
do
	./build/fgprs tailing	$var 1
done

echo quit | nvidia-cuda-mps-control