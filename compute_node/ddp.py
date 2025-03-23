import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import signal
import sys
import os

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
    # master_addr = '128.151.20.130'
    # # master_addr = 'localhost'
    # master_port = '5003'

    # store = dist.TCPStore(
    #     master_addr,
    #     int(master_port),
    #     world_size,
    #     is_master=(rank==0),
    #     use_libuv=False
    # )

    # dist.init_process_group(
    #     "gloo",
    #     store=store,
    #     rank=rank,
    #     world_size=world_size,
    # )

    os.environ['MASTER_ADDR'] = '128.151.20.1300'  # Your Ethernet IP
    os.environ['MASTER_PORT'] = '5003'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Use env:// init method instead of TCPStore
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
    )


    # os.environ['MASTER_ADDR'] = 
    # os.environ['MASTER_PORT'] = '9999'
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '2'
    # dist.init_process_group(
    #     backend="gloo",
    #     init_method="env://",
    #     world_size=2,
    #     rank=rank,
    # )


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    
    # device = torch.device(f"cuda:{rank}")
    device = "cpu"
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10).to(device))
    labels = torch.randn(20,5).to(device)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    import argparse
    world_size = 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current process")
    args = parser.parse_args()

    demo_basic(args.rank, world_size)