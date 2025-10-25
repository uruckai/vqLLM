#!/usr/bin/env python3
"""
Setup script for new fused kernel project
Copies essential files from CodecLLM workspace to a clean new project
"""

import shutil
import os
from pathlib import Path

def setup_fused_kernel_project(source_dir: str = ".", target_dir: str = "fused_kernel_project"):
    """
    Copy essential files for fused kernel implementation to a new project directory

    Args:
        source_dir: Source directory (current workspace)
        target_dir: Target directory for new project
    """

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Essential files to copy
    essential_files = [
        # Core documentation
        "docs/FUSED_KERNEL_IMPLEMENTATION_GUIDE.md",
        "docs/FUSED_KERNEL_TECHNICAL_SPEC.md",
        "docs/FUSED_KERNEL_ANALYSIS.md",
        "docs/SPEED_OPTIMIZATION_GUIDE.md",

        # Working rANS implementation
        "core_rans/CMakeLists.txt",
        "core_rans/c_api.cpp",
        "core_rans/decoder_gpu.cu",
        "core_rans/decoder_host.cpp",
        "core_rans/decoder_host.h",
        "core_rans/encoder_simple.cpp",
        "core_rans/encoder.cpp",
        "core_rans/encoder.h",
        "core_rans/format.h",
        "core_rans/rans.cpp",
        "core_rans/rans.h",

        # Build system and dependencies
        "CMakeLists.txt",
        "requirements.txt",
        "README.md",
        "SETUP.md",
    ]

    # Optional reference files (commented out - copy if needed)
    optional_files = [
        # "python/wcodec/__init__.py",
        # "python/wcodec/bindings.py",
        # "python/wcodec/decoder.py",
        # "python/wcodec/encoder.py",
    ]

    print(f"Setting up fused kernel project in: {target_path}")
    print("="*60)

    # Create target directory structure
    target_path.mkdir(exist_ok=True)
    (target_path / "docs").mkdir(exist_ok=True)
    (target_path / "src").mkdir(exist_ok=True)
    (target_path / "src" / "rans").mkdir(exist_ok=True)

    # Copy essential files
    copied_count = 0
    for file_path in essential_files:
        source_file = source_path / file_path
        if source_file.exists():
            # Map to new structure
            if file_path.startswith("core_rans/"):
                target_file = target_path / "src" / "rans" / file_path.replace("core_rans/", "")
            elif file_path.startswith("docs/"):
                target_file = target_path / file_path
            else:
                target_file = target_path / file_path

            shutil.copy2(source_file, target_file)
            print(f"✓ Copied: {file_path}")
            copied_count += 1
        else:
            print(f"✗ Missing: {file_path}")

    print(f"\n✅ Copied {copied_count} files successfully!")
    print("\n📁 New project structure:")
    print(f"   {target_dir}/")
    print(f"   ├── docs/          # Implementation guides")
    print(f"   ├── src/rans/      # rANS compression library")
    print(f"   ├── CMakeLists.txt # Build configuration")
    print(f"   ├── requirements.txt")
    print(f"   ├── README.md")
    print(f"   └── SETUP.md")

    print("\n🚀 Next steps:")
    print(f"   cd {target_dir}")
    print("   mkdir build && cd build")
    print("   cmake .. -DCMAKE_BUILD_TYPE=Release")
    print("   make -j$(nproc)")
    print("\n📖 Read the implementation guides in docs/ to get started!")
if __name__ == "__main__":
    import sys

    # Default to current directory as source, create "fused_kernel_project" as target
    source = Path.cwd()
    target = "fused_kernel_project"

    if len(sys.argv) > 1:
        target = sys.argv[1]

    setup_fused_kernel_project(source, target)

