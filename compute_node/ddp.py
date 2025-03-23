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
import time
import datetime

def check_connection(host, port, timeout=5):
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, int(port)))
        s.close()
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

def setup(rank, world_size):
    # This IP should be the IP of the machine running rank 0
    master_addr = '10.17.21.244'
    master_port = '5004'
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    print(f"Rank {rank}: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    
    # Test connection before trying to initialize
    if check_connection(master_addr, master_port):
        print(f"Rank {rank}: Successfully connected to {master_addr}:{master_port}")
    else:
        print(f"Rank {rank}: Failed to connect to {master_addr}:{master_port} - check firewall settings")
        if rank != 0:
            print("This is not the master node, so it needs to connect to the master.")
            return False
    
    # Set a shorter timeout for debugging (30 seconds)
    timeout = 30
    print(f"Rank {rank}: Initializing process group with timeout {timeout}s")
    
    try:
        # Try initialization with explicit timeout
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}?use_libuv=False",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout)
        )
        print(f"Rank {rank}: Process group initialized successfully!")
        return True
    except Exception as e:
        print(f"Rank {rank}: Error initializing process group: {e}")
        return False

# Main function with simplified execution for debugging
def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    
    # Just focus on initialization for debugging
    if setup(rank, world_size):
        print(f"Rank {rank}: Setup completed successfully")
        
        try:
            # Test if we can communicate between processes
            if dist.get_rank() == 0:
                tensor = torch.ones(1)
                # Send to rank 1
                dist.send(tensor, dst=1)
                print(f"Rank {rank}: Sent tensor to rank 1")
            else:
                tensor = torch.zeros(1)
                # Receive from rank 0
                dist.recv(tensor, src=0)
                print(f"Rank {rank}: Received tensor from rank 0: {tensor.item()}")
                
            print(f"Rank {rank}: Communication test passed!")
            
        except Exception as e:
            print(f"Rank {rank}: Error during communication test: {e}")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
                print(f"Rank {rank}: Process group destroyed")
    else:
        print(f"Rank {rank}: Setup failed, skipping the rest")
    
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current process")
    parser.add_argument("--world-size", type=int, default=2, help="Total number of processes")
    parser.add_argument("--master-addr", type=str, default="10.17.21.244", help="Master address")
    parser.add_argument("--master-port", type=str, default="5003", help="Master port")
    args = parser.parse_args()
    
    # Get the process ID for debugging
    pid = os.getpid()
    print(f"The PID of the current process is: {pid}")
    
    # Register signal handler for clean exits
    def signal_handler(sig, frame):
        print("Ctrl+C pressed. Cleaning up...")
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the demo
    demo_basic(args.rank, args.world_size)