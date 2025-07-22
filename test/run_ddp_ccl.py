import datetime
from time import perf_counter_ns
import sys
import os
import socket
from mpi4py import MPI
import intel_extension_for_pytorch  # Added Extra
import torch.nn.parallel
import torch.distributed as dist
import oneccl_bindings_for_pytorch

MPI.COMM_WORLD.Barrier()

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
mpi_world_size = MPI.COMM_WORLD.Get_size()
mpi_my_rank = MPI.COMM_WORLD.Get_rank()

if mpi_my_rank == 0:
   master_addr = socket.gethostname()
   sock = socket.socket()
   sock.bind(('', 0))
   # master_port = sock.getsockname()[1] 
   master_port = 2345
else:
   master_addr = None
   master_port = None

master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = str(master_port)

MPI.COMM_WORLD.Barrier()
dist.init_process_group(backend="ccl", init_method='env://', world_size=mpi_world_size, rank=mpi_my_rank, timeout=datetime.timedelta(seconds=3600))
MPI.COMM_WORLD.Barrier()

dist_my_rank = dist.get_rank()
dist_world_size = dist.get_world_size()

def get_default_device():
    if torch.xpu.is_available():
        return torch.device(f"xpu:{dist_my_rank % 12}")
    else:
        return torch.device('cpu')

device = get_default_device()

def test_p2p_communication():
    """Test point-to-point communication operations"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}] Starting P2P test with world_size={world_size}")
    
    if world_size < 2:
        print(f"[Rank {rank}] P2P test requires at least 2 processes, skipping...")
        return
    
    # Create test tensor
    tensor_size = 1024
    test_tensor = torch.ones(tensor_size, dtype=torch.float32, device=device) * rank
    recv_tensor = torch.zeros(tensor_size, dtype=torch.float32, device=device)
    
    # Ring communication pattern
    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1 + world_size) % world_size
    
    print(f"[Rank {rank}] Will send to {send_rank}, receive from {recv_rank}")
    
    # Test 1: Non-blocking isend/irecv
    print(f"[Rank {rank}] Test 1: Non-blocking isend/irecv")
    send_req = dist.isend(test_tensor, dst=send_rank)
    recv_req = dist.irecv(recv_tensor, src=recv_rank)
    
    send_req.wait()
    recv_req.wait()
    
    expected_value = float(recv_rank)
    actual_value = recv_tensor[0].item()
    print(f"[Rank {rank}] Received from {recv_rank}: expected={expected_value}, actual={actual_value}")
    
    # Test 2: Batch P2P operations
    if world_size >= 2:
        print(f"[Rank {rank}] Test 2: Batch P2P operations")
        send_tensor = torch.ones(tensor_size, dtype=torch.float32, device=device) * (rank + 10)
        recv_tensor2 = torch.zeros(tensor_size, dtype=torch.float32, device=device)
        
        ops = []
        ops.append(dist.P2POp(dist.isend, send_tensor, send_rank))
        ops.append(dist.P2POp(dist.irecv, recv_tensor2, recv_rank))
        
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        
        expected_value2 = float(recv_rank + 10)
        actual_value2 = recv_tensor2[0].item()
        print(f"[Rank {rank}] Batch P2P received: expected={expected_value2}, actual={actual_value2}")
    
    print(f"[Rank {rank}] P2P tests completed successfully!")
    MPI.COMM_WORLD.Barrier()



# Original all_reduce test
dim_size = int(int(sys.argv[1]) / 4)
MPI.COMM_WORLD.Barrier()

elapsed1 = []

for _ in range(50):
    x = torch.ones([1, dim_size], dtype=torch.float32).to(device, non_blocking=True)
    # print(x)
    t5 = perf_counter_ns() 
    dist.all_reduce(x, op=dist.ReduceOp.SUM)  # Added Extra op
    MPI.COMM_WORLD.Barrier()
    t6 = perf_counter_ns()
    elapsed1.append(t6 - t5)

if mpi_my_rank == 0:
    for e in elapsed1:
        print(e)

# Run P2P tests first
# test_p2p_communication()