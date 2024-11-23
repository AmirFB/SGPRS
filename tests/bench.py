import torch
import torch.nn as nn
import torch.optim as optim
import time

torch.set_num_threads(16)

# Define the neural network model
class RLModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(RLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x

# Define the RL problem and data
input_size = 60
hidden_size1 = 1280
hidden_size2 = 2560
output_size = 60

# Create an instance of the RLModel
model = RLModel(input_size, hidden_size1, hidden_size2, output_size)

# Define a dummy input tensor for inference
dummy_input = torch.rand((1, input_size))

# Benchmark Inference
n_inference_runs = 1000  # Example: Run inference 1000 times
inference_times = []

for _ in range(n_inference_runs):
    # start_time = time.time()
    with torch.no_grad():
        model(dummy_input)
    # inference_times.append((time.time() - start_time) * 1e6)  # Convert to microseconds

n_inference_runs = 1  # Example: Run inference 1000 times

for _ in range(n_inference_runs):
    start_time = time.time()
    with torch.no_grad():
        model(dummy_input)
    inference_times.append((time.time() - start_time) * 1e6)  # Convert to microseconds

average_inference_time = sum(inference_times) / n_inference_runs
print(f"Average Inference Time: {average_inference_time:.4f} microseconds")

# Define a dummy input and target tensor for training
dummy_input = torch.rand((100, input_size))  # Example batch size of 100
dummy_target = torch.rand((100, output_size))

# Criterion and optimizer for training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Benchmark Training
# n_training_runs = 100  # Example: Run training 100 times
# training_times = []

# for _ in range(n_training_runs):
#     # for epoch in range(100):  # Example of 100 training epochs
# 		# Forward pass
# 		outputs = model(dummy_input)
# 		# start_time = time.time()
# 		loss = criterion(outputs, dummy_target)

# 		# Backward pass and optimization
# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()
		
# 		# training_times.append((time.time() - start_time) * 1e6)  # Convert to microseconds

# n_training_runs = 1  # Example: Run training 100 times

# for _ in range(n_training_runs):
#     # for epoch in range(100):  # Example of 100 training epochs
# 		# Forward pass
# 		outputs = model(dummy_input)
# 		start_time = time.time()
# 		loss = criterion(outputs, dummy_target)

# 		# Backward pass and optimization
# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()
		
# 		training_times.append((time.time() - start_time) * 1e6)  # Convert to microseconds

# average_training_time = sum(training_times) / n_training_runs
# print(f"Average Training Time: {average_training_time:.4f} microseconds")