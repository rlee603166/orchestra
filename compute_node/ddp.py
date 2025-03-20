import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import signal
import sys
import os
import socket

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

def setup(rank, world_size, master_addr, master_port):
    print(f"Setting up process group with master: {master_addr}:{master_port}")
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Different setup for master and worker nodes
    try:
        # Initialize the process group with TCP Store explicitly
        # Setting timeout higher for cross-machine communication
        store = dist.TCPStore(
            master_addr,
            int(master_port),
            world_size,
            is_master=(rank == 0),
            use_libuv=False,
            timeout=datetime.timedelta(seconds=60)
        )
        
        dist.init_process_group(
            "gloo",  # You can change to "nccl" if using GPUs
            store=store,
            rank=rank,
            world_size=world_size,
        )
        print(f"Successfully initialized process group for rank {rank}")
    except Exception as e:
        print(f"Error initializing process group: {e}")
        raise

def cleanup():
    if dist.is_initialized():
        print("Destroying process group")
        dist.destroy_process_group()

def demo_basic(rank, world_size, master_addr, master_port):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size, master_addr, master_port)
    
    # Use CPU if CUDA is not available
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
        
    model = ToyModel().to(device)
    
    # Modify DDP setup for CPU if needed
    if torch.cuda.is_available():
        ddp_model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    else:
        ddp_model = DDP(model)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10).to(device))
    labels = torch.randn(20, 5).to(device)
    loss = loss_fn(outputs, labels)
    loss.backward()  # Fixed: changed from backwards() to backward()
    optimizer.step()
    
    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current process")
    parser.add_argument("--world-size", type=int, default=2, help="Total number of processes")
    parser.add_argument("--master-addr", type=str, default="128.151.20.178", 
                        help="Master node IP address (use actual IP when running across machines)")
    parser.add_argument("--master-port", type=str, default="8081", 
                        help="Master node port")
    
    args = parser.parse_args()
    
    print(f"Starting process with rank {args.rank} out of {args.world_size}")
    print(f"Using master address: {args.master_addr}:{args.master_port}")
    
    try:
        demo_basic(args.rank, args.world_size, args.master_addr, args.master_port)
    except Exception as e:
        print(f"Error in demo_basic: {e}")
        # Make sure to clean up even if there's an error
        cleanup()
        sys.exit(1)