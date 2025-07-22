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



def test_p2p_communication_fixed():
    """fixed P2P communication"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # initialization
    warmup = torch.ones(1, device=device)
    dist.all_reduce(warmup)
    dist.barrier()
    
    print(f"[Rank {rank}] Starting P2P test with world_size={world_size}")
    
    if world_size < 2:
        print(f"[Rank {rank}] P2P test requires at least 2 processes, skipping...")
        return
    
    tensor_size = 1024
    test_tensor = torch.ones(tensor_size, dtype=torch.float32, device=device) * rank
    recv_tensor = torch.zeros(tensor_size, dtype=torch.float32, device=device)
    
    # fix: avoid all ranks doing ring communication at the same time
    # use safer communication pattern: even rank first send, odd rank first recv
    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1 + world_size) % world_size
    
    print(f"[Rank {rank}] Will send to {send_rank}, receive from {recv_rank}")
    
    # Test 1: improved non-blocking communication
    print(f"[Rank {rank}] Test 1: Fixed non-blocking isend/irecv")
    
    if rank % 2 == 0:  # even rank first send
        send_req = dist.isend(test_tensor, dst=send_rank)
        recv_req = dist.irecv(recv_tensor, src=recv_rank)
    else:  # odd rank first recv
        recv_req = dist.irecv(recv_tensor, src=recv_rank)
        send_req = dist.isend(test_tensor, dst=send_rank)
    
    # wait for completion
    send_req.wait()
    recv_req.wait()
    
    expected_value = float(recv_rank)
    actual_value = recv_tensor[0].item()
    print(f"[Rank {rank}] Received from {recv_rank}: expected={expected_value}, actual={actual_value}")
    
    # barrier
    dist.barrier()
    
    print(f"[Rank {rank}] P2P tests completed successfully!")


def test_p2p_communication_minimal_fix():
    """minimal fix for P2P communication"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # necessary initialization
    warmup = torch.ones(1, device=device)
    dist.all_reduce(warmup)
    dist.barrier()
    
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
    
    # Test 1: key fix: change the order of isend/irecv based on rank parity
    print(f"[Rank {rank}] Test 1: Non-blocking isend/irecv")
    
    # key fix: change the order of isend/irecv based on rank parity
    if rank % 2 == 0:
        send_req = dist.isend(test_tensor, dst=send_rank)
        recv_req = dist.irecv(recv_tensor, src=recv_rank)
    else:
        recv_req = dist.irecv(recv_tensor, src=recv_rank)
        send_req = dist.isend(test_tensor, dst=send_rank)
    
    send_req.wait()
    recv_req.wait()
    
    expected_value = float(recv_rank)
    actual_value = recv_tensor[0].item()
    print(f"[Rank {rank}] Received from {recv_rank}: expected={expected_value}, actual={actual_value}")
    
    # add barrier
    dist.barrier()
    
    # Test 2: simplified P2P operations
    if world_size >= 2:
        print(f"[Rank {rank}] Test 2: Simplified P2P operations")
        send_tensor = torch.ones(tensor_size, dtype=torch.float32, device=device) * (rank + 10)
        recv_tensor2 = torch.zeros(tensor_size, dtype=torch.float32, device=device)
        
        # replace batch operations with simple operations
        if rank % 2 == 0:
            send_req2 = dist.isend(send_tensor, dst=send_rank)
            recv_req2 = dist.irecv(recv_tensor2, src=recv_rank)
        else:
            recv_req2 = dist.irecv(recv_tensor2, src=recv_rank)
            send_req2 = dist.isend(send_tensor, dst=send_rank)
        
        send_req2.wait()
        recv_req2.wait()
        
        expected_value2 = float(recv_rank + 10)
        actual_value2 = recv_tensor2[0].item()
        print(f"[Rank {rank}] Simplified P2P received: expected={expected_value2}, actual={actual_value2}")
    
    # final barrier
    dist.barrier()
    print(f"[Rank {rank}] P2P tests completed successfully!")



def debug_p2p_issues():
    """系统性地诊断 P2P 通信问题"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 初始化
    warmup = torch.ones(1, device=device)
    dist.all_reduce(warmup)
    dist.barrier()
    
    print(f"[Rank {rank}] Starting P2P debugging with world_size={world_size}")
    
    if world_size < 2:
        print(f"[Rank {rank}] Need at least 2 processes")
        return
    
    tensor_size = 1024
    
    # 测试1: 简单的非环形通信 + batch_isend_irecv
    def test1_batch_non_ring():
        print(f"[Rank {rank}] Test 1: batch_isend_irecv with NON-RING communication")
        try:
            if rank == 0:
                # Rank 0 发送给 Rank 1，接收来自 Rank 1
                send_tensor = torch.ones(tensor_size, device=device) * 100
                recv_tensor = torch.zeros(tensor_size, device=device)
                
                ops = []
                ops.append(dist.P2POp(dist.isend, send_tensor, 1))
                ops.append(dist.P2POp(dist.irecv, recv_tensor, 1))
                
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
                    
                print(f"[Rank {rank}] Test 1: SUCCESS - received {recv_tensor[0].item()}")
                
            elif rank == 1:
                # Rank 1 接收来自 Rank 0，发送给 Rank 0
                send_tensor = torch.ones(tensor_size, device=device) * 200
                recv_tensor = torch.zeros(tensor_size, device=device)
                
                ops = []
                ops.append(dist.P2POp(dist.irecv, recv_tensor, 0))  # 先接收
                ops.append(dist.P2POp(dist.isend, send_tensor, 0))
                
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
                    
                print(f"[Rank {rank}] Test 1: SUCCESS - received {recv_tensor[0].item()}")
            
            dist.barrier()
            return True
            
        except Exception as e:
            print(f"[Rank {rank}] Test 1: FAILED - {e}")
            return False
    
    # 测试2: 环形通信 + 简单 isend/irecv (非batch)
    def test2_ring_simple():
        print(f"[Rank {rank}] Test 2: RING communication with simple isend/irecv")
        try:
            send_tensor = torch.ones(tensor_size, device=device) * rank
            recv_tensor = torch.zeros(tensor_size, device=device)
            
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1 + world_size) % world_size
            
            # 使用简单的 isend/irecv，但保持环形模式
            if rank % 2 == 0:
                send_req = dist.isend(send_tensor, dst=send_rank)
                recv_req = dist.irecv(recv_tensor, src=recv_rank)
            else:
                recv_req = dist.irecv(recv_tensor, src=recv_rank)
                send_req = dist.isend(send_tensor, dst=send_rank)
            
            send_req.wait()
            recv_req.wait()
            
            expected = float(recv_rank)
            actual = recv_tensor[0].item()
            print(f"[Rank {rank}] Test 2: SUCCESS - expected {expected}, got {actual}")
            
            dist.barrier()
            return True
            
        except Exception as e:
            print(f"[Rank {rank}] Test 2: FAILED - {e}")
            return False
    
    # 测试3: 环形通信 + batch_isend_irecv (原始问题)
    def test3_ring_batch():
        print(f"[Rank {rank}] Test 3: RING communication with batch_isend_irecv")
        try:
            send_tensor = torch.ones(tensor_size, device=device) * rank
            recv_tensor = torch.zeros(tensor_size, device=device)
            
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1 + world_size) % world_size
            
            # 这是原始的问题组合：环形 + batch
            ops = []
            ops.append(dist.P2POp(dist.isend, send_tensor, send_rank))
            ops.append(dist.P2POp(dist.irecv, recv_tensor, recv_rank))
            
            print(f"[Rank {rank}] Test 3: About to call batch_isend_irecv...")
            reqs = dist.batch_isend_irecv(ops)
            print(f"[Rank {rank}] Test 3: batch_isend_irecv returned, waiting...")
            
            for req in reqs:
                req.wait()
            
            expected = float(recv_rank)
            actual = recv_tensor[0].item()
            print(f"[Rank {rank}] Test 3: SUCCESS - expected {expected}, got {actual}")
            
            dist.barrier()
            return True
            
        except Exception as e:
            print(f"[Rank {rank}] Test 3: FAILED - {e}")
            return False
    
    # 测试4: 检查 batch_isend_irecv 在其他模式下的行为
    def test4_batch_variations():
        print(f"[Rank {rank}] Test 4: batch_isend_irecv variations")
        try:
            if world_size < 4:
                print(f"[Rank {rank}] Test 4: Need at least 4 processes for this test")
                return True
            
            # 使用跳跃模式而非环形模式
            if rank < world_size // 2:
                # 前半部分进程
                partner = rank + world_size // 2
                send_tensor = torch.ones(tensor_size, device=device) * rank
                recv_tensor = torch.zeros(tensor_size, device=device)
                
                ops = []
                ops.append(dist.P2POp(dist.isend, send_tensor, partner))
                ops.append(dist.P2POp(dist.irecv, recv_tensor, partner))
                
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
                    
                print(f"[Rank {rank}] Test 4: SUCCESS - communicated with {partner}")
                
            else:
                # 后半部分进程
                partner = rank - world_size // 2
                send_tensor = torch.ones(tensor_size, device=device) * rank
                recv_tensor = torch.zeros(tensor_size, device=device)
                
                ops = []
                ops.append(dist.P2POp(dist.irecv, recv_tensor, partner))  # 先接收
                ops.append(dist.P2POp(dist.isend, send_tensor, partner))
                
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
                    
                print(f"[Rank {rank}] Test 4: SUCCESS - communicated with {partner}")
            
            dist.barrier()
            return True
            
        except Exception as e:
            print(f"[Rank {rank}] Test 4: FAILED - {e}")
            return False
    
    # 运行所有测试
    results = {}
    
    print(f"\n[Rank {rank}] ========== 开始诊断测试 ==========")
    
    # 测试1: 检查 batch_isend_irecv 在非环形模式下是否工作
    results['batch_non_ring'] = test1_batch_non_ring()
    
    if rank == 0:
        print(f"\n========== 测试1结果分析 ==========")
        if results['batch_non_ring']:
            print("✓ batch_isend_irecv 在非环形通信下工作正常")
        else:
            print("✗ batch_isend_irecv 本身有问题")
    
    dist.barrier()
    
    # 测试2: 检查环形通信在简单模式下是否工作
    results['ring_simple'] = test2_ring_simple()
    
    if rank == 0:
        print(f"\n========== 测试2结果分析 ==========")
        if results['ring_simple']:
            print("✓ 环形通信模式本身工作正常")
        else:
            print("✗ 环形通信模式有问题")
    
    dist.barrier()
    
    # 测试3: 原始问题组合
    results['ring_batch'] = test3_ring_batch()
    
    if rank == 0:
        print(f"\n========== 测试3结果分析 ==========")
        if results['ring_batch']:
            print("✓ 环形通信 + batch_isend_irecv 工作正常（这很意外！）")
        else:
            print("✗ 环形通信 + batch_isend_irecv 确实有问题")
    
    dist.barrier()
    
    # # 测试4: 其他batch模式
    results['batch_variations'] = test4_batch_variations()
    
    if rank == 0:
        print(f"\n========== 最终诊断结果 ==========")
        
        if not results['batch_non_ring']:
            print("🔍 结论: batch_isend_irecv 函数本身有问题")
        elif not results['ring_simple']:
            print("🔍 结论: 环形通信模式在OneCCL中有问题")
        elif not results['ring_batch'] and results['batch_non_ring'] and results['ring_simple']:
            print("🔍 结论: batch_isend_irecv + 环形通信的组合导致死锁")
            print("   - batch_isend_irecv 单独工作正常")
            print("   - 环形通信单独工作正常") 
            print("   - 但两者结合时出现问题")
        elif results['ring_batch']:
            print("🔍 结论: 所有基本模式都工作，问题可能在原始代码的其他细节")
        else:
            print("🔍 结论: 需要更详细的分析")
            
        print(f"\n测试结果摘要:")
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {test_name}: {status}")
    
    dist.barrier()
    print(f"[Rank {rank}] 诊断完成")


# 调用诊断函数
debug_p2p_issues()




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
        # print(e)
        pass

# MPI.COMM_WORLD.Barrier()
# Run P2P tests first
# test_p2p_communication()
# simple_p2p_test()
# test_p2p_communication_fixed()
# test_p2p_communication_staged()
# test_p2p_communication_minimal_fix()

debug_p2p_issues()


