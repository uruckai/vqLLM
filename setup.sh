#!/bin/bash
#
# CodecLLM Setup Script
# Automated installation of all dependencies and build
#
# Usage:
#   ./setup.sh              # Full setup
#   ./setup.sh --verify     # Verify existing installation
#   ./setup.sh --rebuild    # Rebuild codec library only
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_step() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if running as root (needed for apt-get)
if [ "$EUID" -ne 0 ]; then 
    print_warning "Not running as root. Some steps may require sudo."
fi

# Parse arguments
VERIFY_ONLY=false
REBUILD_ONLY=false

for arg in "$@"; do
    case $arg in
        --verify)
            VERIFY_ONLY=true
            ;;
        --rebuild)
            REBUILD_ONLY=true
            ;;
        --help)
            echo "CodecLLM Setup Script"
            echo ""
            echo "Usage:"
            echo "  ./setup.sh              Full setup (default)"
            echo "  ./setup.sh --verify     Verify existing installation"
            echo "  ./setup.sh --rebuild    Rebuild codec library only"
            echo "  ./setup.sh --help       Show this help"
            exit 0
            ;;
    esac
done

echo "=================================================================="
echo "          CodecLLM Setup - GPU-Accelerated LLM Compression"
echo "=================================================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Verification function
verify_installation() {
    print_step "Verifying installation..."
    
    local errors=0
    
    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_success "nvidia-smi found"
    else
        print_error "nvidia-smi not found - GPU not available"
        errors=$((errors + 1))
    fi
    
    # Check nvCOMP
    if ldconfig -p | grep -q libnvcomp; then
        print_success "nvCOMP library found"
    else
        print_error "nvCOMP not found in library path"
        errors=$((errors + 1))
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        print_success "Python3 found: $(python3 --version)"
    else
        print_error "Python3 not found"
        errors=$((errors + 1))
    fi
    
    # Check Python packages
    if python3 -c "import torch" 2>/dev/null; then
        print_success "PyTorch installed"
    else
        print_error "PyTorch not installed"
        errors=$((errors + 1))
    fi
    
    if python3 -c "import transformers" 2>/dev/null; then
        print_success "Transformers installed"
    else
        print_error "Transformers not installed"
        errors=$((errors + 1))
    fi
    
    # Check codec library
    if [ -f "$SCRIPT_DIR/core/build/libcodec_core.so" ]; then
        print_success "Codec library built"
    else
        print_error "Codec library not found at $SCRIPT_DIR/core/build/libcodec_core.so"
        errors=$((errors + 1))
    fi
    
    # Check if codec library loads
    if [ -f "$SCRIPT_DIR/core/build/libcodec_core.so" ]; then
        if python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR/core'); from bindings_zstd import ZstdEncoder, ZstdGPUDecoder" 2>/dev/null; then
            print_success "Codec library loads successfully"
        else
            print_error "Codec library found but fails to load"
            errors=$((errors + 1))
        fi
    fi
    
    echo ""
    if [ $errors -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed!${NC}"
        return 0
    else
        echo -e "${RED}✗ Found $errors issue(s)${NC}"
        return 1
    fi
}

# Rebuild function
rebuild_codec() {
    print_step "Rebuilding codec library..."
    
    cd "$SCRIPT_DIR/core"
    mkdir -p build
    cd build
    
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make clean
    make -j$(nproc)
    
    if [ -f "libcodec_core.so" ]; then
        print_success "Codec library rebuilt successfully"
        ls -lh libcodec_core.so
    else
        print_error "Build failed - libcodec_core.so not created"
        exit 1
    fi
}

# Handle special modes
if [ "$VERIFY_ONLY" = true ]; then
    verify_installation
    exit $?
fi

if [ "$REBUILD_ONLY" = true ]; then
    rebuild_codec
    exit $?
fi

# Full setup
print_step "[1/6] Checking system packages..."

if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y git cmake build-essential wget > /dev/null 2>&1
    print_success "System packages installed"
else
    print_warning "apt-get not available - assuming packages are installed"
fi

print_step "[2/6] Installing nvCOMP 3.0.6..."

# Check if already installed
if ldconfig -p | grep -q libnvcomp; then
    print_success "nvCOMP already installed (skipping)"
else
    cd /tmp
    
    if [ -f "nvcomp_3.0.6_x86_64_12.x.tgz" ]; then
        print_success "nvCOMP archive already downloaded"
    else
        print_step "Downloading nvCOMP (this may take a minute)..."
        wget -q --show-progress https://developer.download.nvidia.com/compute/nvcomp/3.0.6/local_installers/nvcomp_3.0.6_x86_64_12.x.tgz
    fi
    
    print_step "Extracting and installing..."
    tar -xzf nvcomp_3.0.6_x86_64_12.x.tgz
    
    # The archive extracts directly (no subdirectory)
    if [ -d "include" ] && [ -d "lib" ]; then
        cp -r include/* /usr/local/include/
        cp -r lib/* /usr/local/lib/
    else
        print_error "nvCOMP structure not as expected"
        ls -la
        exit 1
    fi
    
    ldconfig
    
    # Clean up
    rm -rf include lib bin doc *.md LICENSE NOTICE nvcomp_3.0.6_x86_64_12.x.tgz
    
    print_success "nvCOMP installed"
fi

print_step "[3/6] Installing Python packages..."

pip install -q torch transformers accelerate numpy hf-transfer
print_success "Python packages installed"

print_step "[4/6] Building codec library..."

cd "$SCRIPT_DIR/core"
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null
make -j$(nproc) 2>&1 | grep -v "warning:" || true

if [ -f "libcodec_core.so" ]; then
    print_success "Codec library built successfully"
    ls -lh libcodec_core.so
else
    print_error "Build failed - see errors above"
    exit 1
fi

print_step "[5/6] Setting environment..."

# Disable HF transfer by default (causes errors if not installed)
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "export HF_HUB_ENABLE_HF_TRANSFER=0" >> ~/.bashrc

print_success "Environment configured"

print_step "[6/6] Verifying installation..."
echo ""

verify_installation

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================================="
    echo -e "${GREEN}          Setup Complete!${NC}"
    echo "=================================================================="
    echo ""
    echo "Quick Start:"
    echo "  cd $SCRIPT_DIR/core"
    echo "  python3 test_fp32_kv_cache.py"
    echo ""
    echo "For cleaner output:"
    echo "  python3 test_fp32_kv_cache.py 2>&1 | grep -vE \"ENCODER|DECODER\""
    echo ""
    echo "To verify anytime:"
    echo "  ./setup.sh --verify"
    echo ""
else
    echo ""
    echo "=================================================================="
    echo -e "${RED}          Setup Incomplete${NC}"
    echo "=================================================================="
    echo ""
    echo "Some checks failed. Please review errors above."
    echo "For manual setup, see SETUP.md"
    exit 1
fi

