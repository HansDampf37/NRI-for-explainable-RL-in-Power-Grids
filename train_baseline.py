from baselines.train_stable_dqn_baseline import main
import torch

if not torch.cuda.is_available():
    print("CUDA is not available")
    raise ValueError("CUDA is not available")

main()
