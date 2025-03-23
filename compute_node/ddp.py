import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import signal
import sys
import os
import time

pid = os.getpid()
print(f"The PID of the current process is: {pid}")

def signal_handler(sig, frame):
    print("Ctrl+C pressed. Cleaning up...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def setup(rank, world_size):
    # This IP should be the IP of the machine running rank 0
    master_addr = '10.17.21.244'
    master_port = '5003'
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Print environment for debugging
    print(f"Rank {rank}: MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
    
    # Extended timeout for multi-machine setup (30 minutes)
    timeout = 1800
    
    print(f"Rank {rank}: Initializing process group")
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}?use_libuv=False",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout)
        )
        print(f"Rank {rank}: Process group initialized successfully")
    except Exception as e:
        print(f"Rank {rank}: Error initializing process group: {e}")
        raise

def cleanup():
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: Destroying process group")
        dist.destroy_process_group()

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    
    try:
        setup(rank, world_size)
        print(f"Rank {rank}: Setup completed")
        
        # Add barrier to ensure all processes are ready
        dist.barrier()
        print(f"Rank {rank}: Passed barrier")
        
        device = torch.device("cpu")
        model = ToyModel().to(device)
        ddp_model = DDP(model)
        
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        
        # Small synchronization to ensure all processes have built the model
        dist.barrier()
        print(f"Rank {rank}: Model set up complete")
        
        optimizer.zero_grad()
        inputs = torch.randn(20, 10).to(device)
        outputs = ddp_model(inputs)
        labels = torch.randn(20, 5).to(device)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Final sync point
        dist.barrier()
        print(f"Rank {rank}: Training step complete")
        
    except Exception as e:
        print(f"Rank {rank}: Error during execution: {e}")
    finally:
        cleanup()
    
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    import argparse
    import datetime
    
    world_size = 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current process")
    parser.add_argument("--world-size", type=int, default=2, help="Total number of processes")
    args = parser.parse_args()
    
    # Update world_size from arguments
    world_size = args.world_size
    
    demo_basic(args.rank, world_size)