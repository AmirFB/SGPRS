import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34
import time

torch.backends.cudnn.enabled = False

def benchmark_resnet(model, input_size, num_iterations):
    # Move the model to CUDA
    model = model.cuda()

    # Dummy input tensor
    dummy_input = torch.randn(1, 3, input_size, input_size).cuda()

    # Set the model to evaluation mode
    model.eval()

    # Warm-up
    for _ in range(5):
        model(dummy_input)

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        model(dummy_input)
    end_time = time.time()

    # Calculate frames per second (fps)
    fps = num_iterations / (end_time - start_time)

    return fps

# ResNet-18 benchmark
with torch.no_grad():
	resnet18_model = resnet18()
	resnet18_fps = benchmark_resnet(resnet18_model, 224, 100)
	print(f"ResNet-18 - Input Size: 224x224, FPS: {resnet18_fps}")

	# ResNet-34 benchmark
	resnet34_model = resnet34()
	resnet34_fps = benchmark_resnet(resnet34_model, 224, 100)
	print(f"ResNet-34 - Input Size: 224x224, FPS: {resnet34_fps}")
