#!/usr/bin/env python3
"""
Compile MLIR to IREE bytecode (VMFB) following the IREE.gd documentation.

This script compiles step_model_from_onnx.mlir to bytecode for different backends:
- CPU (LLVM)
- Vulkan (for cross-platform)
- Metal (for Apple devices)

Based on: https://iree-gd.github.io/iree.gd.docs/iree.gd/
"""

import os
import subprocess
import shutil
from pathlib import Path


def compile_mlir_to_vmfb(mlir_path: str, target_backend: str, output_dir: str = ".exports"):
    """
    Compile MLIR to VMFB using iree-compile following IREE.gd guidelines.
    
    Args:
        mlir_path: Path to the MLIR file
        target_backend: Target backend ('cpu', 'vulkan', 'metal')
        output_dir: Output directory for VMFB files
    """
    
    if not mlir_path or not os.path.exists(mlir_path):
        print(f"ERROR: MLIR file not found: {mlir_path}")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate output filename based on backend
    if target_backend == 'vulkan':
        vmfb_path = output_path / "step_model.vulkan.vmfb"
    elif target_backend == 'metal':
        vmfb_path = output_path / "step_model.metal.vmfb"
    else:  # cpu
        vmfb_path = output_path / "step_model.cpu.vmfb"
    
    # Check if iree-compile is available
    if not shutil.which("iree-compile"):
        print("ERROR: iree-compile not found in PATH")
        print("   Install with: pip install iree-compiler")
        return None
    
    try:
        # Base command - use absolute paths
        abs_mlir_path = os.path.abspath(mlir_path)
        abs_vmfb_path = os.path.abspath(vmfb_path)
        cmd = ["iree-compile", abs_mlir_path, "-o", abs_vmfb_path]
        
        # Add backend-specific flags based on IREE.gd documentation
        if target_backend == 'vulkan':
            cmd.extend([
                "--iree-hal-target-backends=vulkan-spirv"
            ])
            print(f"Compiling for Vulkan backend (Windows, Linux, Android)...")
            
        elif target_backend == 'metal':
            cmd.extend([
                "--iree-hal-target-backends=metal-spirv"
            ])
            print(f"Compiling for Metal backend (Apple devices)...")
            
        else:  # cpu
            cmd.extend([
                "--iree-hal-target-backends=llvm-cpu"
            ])
            print(f"Compiling for CPU backend (LLVM)...")
        
        # Since this MLIR came from ONNX via iree-import-onnx, we might need to specify input type
        # The documentation suggests using --iree-input-type=tosa for TensorFlow Lite imports
        # For ONNX imports, we may need different settings, but let's try without first
        
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            file_size = vmfb_path.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"SUCCESS: VMFB for {target_backend} saved to: {vmfb_path} ({file_size:.2f} MB)")
            return str(vmfb_path)
        else:
            print(f"ERROR: iree-compile failed for {target_backend}:")
            print(f"   stderr: {result.stderr}")
            print(f"   stdout: {result.stdout}")
            return None
            
    except Exception as e:
        print(f"ERROR: Error during VMFB compilation for {target_backend}: {e}")
        return None


def generate_bytecode_info_dump(vmfb_path: str):
    """
    Generate information dump for the bytecode to inspect input/output formats.
    
    This is crucial for understanding how to interface with the model in Godot.
    """
    if not vmfb_path or not os.path.exists(vmfb_path):
        print(f"ERROR: VMFB file not found: {vmfb_path}")
        return None
    
    # Check if iree-dump-module is available
    if not shutil.which("iree-dump-module"):
        print("ERROR: iree-dump-module not found in PATH")
        print("   Install with: pip install iree-tools")
        return None
    
    dump_path = Path(vmfb_path).with_suffix('.dump.log')
    
    try:
        print(f"Generating bytecode information dump...")
        cmd = ["iree-dump-module", vmfb_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            with open(dump_path, 'w') as f:
                f.write(result.stdout)
            print(f"SUCCESS: Bytecode info dump saved to: {dump_path}")
            
            # Print key information about exported functions
            print("\nExported Functions:")
            for line in result.stdout.split('\n'):
                if 'main(' in line or 'iree.abi.declaration' in line:
                    print(f"   {line.strip()}")
            
            return str(dump_path)
        else:
            print(f"ERROR: iree-dump-module failed:")
            print(f"   stderr: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"ERROR: Error generating info dump: {e}")
        return None


def main():
    """Main function to compile MLIR to bytecode for all backends."""
    
    # Path to the MLIR file
    mlir_file = ".exports/step_model_from_onnx.mlir"
    output_dir = ".exports"
    
    if not os.path.exists(mlir_file):
        print(f"ERROR: MLIR file not found: {mlir_file}")
        print("   Run convert_mlir.py first to generate the MLIR file.")
        return
    
    print("Compiling MLIR to IREE bytecode (VMFB) for different backends")
    print(f"Input: {mlir_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Compile for different backends
    backends = ['cpu', 'vulkan', 'metal']
    compiled_files = []
    
    for backend in backends:
        print(f"\n{'='*50}")
        vmfb_path = compile_mlir_to_vmfb(mlir_file, backend, output_dir)
        if vmfb_path:
            compiled_files.append((backend, vmfb_path))
    
    # Generate information dumps for the first successful compilation
    if compiled_files:
        print(f"\n{'='*50}")
        print("Generating bytecode information dump...")
        _, first_vmfb = compiled_files[0]
        dump_path = generate_bytecode_info_dump(first_vmfb)
    
    # Summary
    print(f"\n{'='*50}")
    print("Compilation Summary:")
    print(f"Successfully compiled {len(compiled_files)} backends:")
    
    for backend, vmfb_path in compiled_files:
        file_size = Path(vmfb_path).stat().st_size / (1024 * 1024)
        print(f"   - {backend:>6}: {Path(vmfb_path).name} ({file_size:.2f} MB)")
    
    if len(compiled_files) < len(backends):
        failed = len(backends) - len(compiled_files)
        print(f"Failed to compile {failed} backends")
    


if __name__ == "__main__":
    main()
