#!/bin/bash
# Comprehensive test script for Intel GPU Ring Flash Attention

echo "🚀 Running Intel GPU Ring Flash Attention Test Suite"
echo "=================================================="

# Check if Intel GPU is available
python3 -c "import torch; import intel_extension_for_pytorch; assert torch.xpu.is_available()" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Intel GPU not available or IPEX not installed"
    echo "   Please ensure:"
    echo "   1. Intel GPU drivers are installed"
    echo "   2. Intel Extension for PyTorch is installed"
    echo "   3. You're running on a system with Intel GPU"
    exit 1
fi

echo "✅ Intel GPU environment detected"
echo ""

# Run basic single GPU tests
echo "1️⃣ Running basic Intel GPU tests..."
echo "======================================"
python3 test_intel_gpu.py
BASIC_RESULT=$?
echo ""

# Run comprehensive single GPU tests
echo "2️⃣ Running comprehensive single GPU tests..."
echo "=============================================="
python3 test_intel_ring_flash_attn.py
SINGLE_RESULT=$?
echo ""

# Run variable length tests
echo "3️⃣ Running variable length tests..."
echo "====================================="
python3 test_intel_ring_flash_attn_varlen.py
VARLEN_RESULT=$?
echo ""

# Run distributed tests if multiple GPUs available
GPU_COUNT=$(python3 -c "import torch; print(torch.xpu.device_count())")
if [ $GPU_COUNT -gt 1 ]; then
    echo "4️⃣ Running distributed tests (${GPU_COUNT} GPUs detected)..."
    echo "================================================"
    
    # Test with 2 GPUs
    echo "Testing with 2 GPUs..."
    torchrun --nproc_per_node=2 test_intel_ring_flash_attn.py
    DIST2_RESULT=$?
    
    # Test with all GPUs if more than 2
    if [ $GPU_COUNT -gt 2 ]; then
        echo "Testing with all ${GPU_COUNT} GPUs..."
        torchrun --nproc_per_node=${GPU_COUNT} test_intel_ring_flash_attn.py
        DISTALL_RESULT=$?
    else
        DISTALL_RESULT=0
    fi
else
    echo "⚠️  Only 1 Intel GPU detected, skipping distributed tests"
    DIST2_RESULT=0
    DISTALL_RESULT=0
fi

# Run oneCCL tests if available
echo ""
echo "5️⃣ Testing oneCCL backend..."
echo "============================"
python3 test_oneccl.py
ONECCL_RESULT=$?

# Summary
echo ""
echo "=================================================="
echo "📊 TEST SUMMARY"
echo "=================================================="

TOTAL_TESTS=6
PASSED_TESTS=0

# Function to print test result
print_result() {
    if [ $2 -eq 0 ]; then
        echo "✅ $1: PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "❌ $1: FAILED"
    fi
}

print_result "Basic Intel GPU Tests" $BASIC_RESULT
print_result "Single GPU Comprehensive Tests" $SINGLE_RESULT
print_result "Variable Length Tests" $VARLEN_RESULT
print_result "Distributed Tests (2 GPUs)" $DIST2_RESULT
print_result "Distributed Tests (All GPUs)" $DISTALL_RESULT
print_result "oneCCL Backend Tests" $ONECCL_RESULT

echo ""
echo "Total: ${PASSED_TESTS}/${TOTAL_TESTS} test suites passed"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo "🎉 All tests passed! Intel GPU Ring Flash Attention is working correctly!"
    exit 0
else
    echo "⚠️  Some tests failed. Please check the output above for details."
    exit 1
fi