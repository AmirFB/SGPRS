import torch
import torch.nn as nn
import time
import threading
from torchvision import models

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Use models.resnet18 directly instead of loading through torch.hub
        self.resnet = models.resnet18(pretrained=False)

    def forward(self, x):
        return self.resnet(x)

# Function to run the inference asynchronously
def async_inference(model, input_data, stream, num_iterations, result_dict, run_multiple_iterations):
    model = ResNet()
    model = model.cuda()
    input_data = input_data.cuda()

    # Warm-up GPU
    with torch.cuda.stream(stream):
        warmup_input = torch.randn_like(input_data)
        model(warmup_input)
        stream.synchronize()

    start_time = time.time()
    if run_multiple_iterations:
        with torch.cuda.stream(stream):
            for _ in range(num_iterations):
                output = model(input_data)
                stream.synchronize()  # Synchronize at the end of each iteration
    else:
        with torch.cuda.stream(stream):
            output = model(input_data)
        stream.synchronize()

    average_time = (time.time() - start_time) / num_iterations * 1000 if run_multiple_iterations else 0
    result_dict['average_time'] = average_time

# Function to run the benchmark with different stream priorities
def stream_priority_benchmark(model_class, num_low_priority=16, num_high_priority=16, num_iterations=10):
    # Create the ResNet model
    resnet_model = model_class()

    # Define input size (adjust as needed)
    input_size = (5, 3, 224, 224)

    # Create CUDA streams for low and high priority tasks
    low_priority_streams = [torch.cuda.Stream(priority=0) for _ in range(num_low_priority)]
    high_priority_streams = [torch.cuda.Stream(priority=0) for _ in range(num_high_priority)]

    # Create dictionaries to store results
    low_priority_results = [{} for _ in range(num_low_priority)]
    high_priority_results = [{} for _ in range(num_high_priority)]

    # Create threads for low priority tasks
    low_priority_threads = [threading.Thread(
        target=async_inference,
        args=(resnet_model, torch.randn(input_size).cuda(), low_priority_streams[i], num_iterations, low_priority_results[i], i == (num_low_priority - 6))
    ) for i in range(num_low_priority)]

    # Create threads for high priority tasks
    high_priority_threads = [threading.Thread(
        target=async_inference,
        args=(resnet_model, torch.randn(input_size).cuda(), high_priority_streams[i], num_iterations, high_priority_results[i], i == (num_low_priority - 6))
    ) for i in range(num_high_priority)]

    torch.cuda.profiler.start()

    # Start the threads
    for thread in low_priority_threads:
        thread.start()

    for thread in high_priority_threads:
        thread.start()

    # Wait for the threads to finish
    for thread in low_priority_threads:
        thread.join()

    for thread in high_priority_threads:
        thread.join()

    # Print the results
    print(f"Low Priority Tasks:")
    for i in range(num_low_priority):
        print(f"Task {i + 1} Average Time: {low_priority_results[i]['average_time']:.4f} ms")

    print(f"\nHigh Priority Tasks:")
    for i in range(num_high_priority):
        print(f"Task {i + 1} Average Time: {high_priority_results[i]['average_time']:.4f} ms")

# Run the benchmark
stream_priority_benchmark(ResNet, num_low_priority=12, num_high_priority=12)


torch.cuda.profiler.stop()