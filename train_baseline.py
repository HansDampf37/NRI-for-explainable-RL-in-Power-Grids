from baselines.train_stable_baseline import main
import torch

if not torch.cuda.is_available():
    print("CUDA is not available")
    raise ValueError("CUDA is not available")

print("Available GPUs")
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

main()
