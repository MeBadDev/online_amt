"""
MLIR Export Script for Audio Transcription Model

This script exports PyTorch models to MLIR and compiles them to VMFB for different backends.

LSTM Warning Explanation:
The ONNX export may show a warning about LSTM batch sizes. This is normal and expected when:
- Using dynamic batch dimensions with LSTM layers
- The hidden states (h0/c0) dimensions could vary at runtime

Solutions implemented:
1. export_to_onnx_detailed: Uses only time-dynamic axes, fixed batch=1
2. export_to_onnx_batch_1_fixed: Completely fixed dimensions, no warnings

For audio transcription, batch=1 is typically sufficient since inference 
is usually done on single audio streams.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import tempfile
from pathlib import Path

# Import model loading function
from transcribe import load_model


def simple_export_to_mlir(model, output_dir: str = "exports"):
    """Simple export using torch.jit.trace and manual MLIR generation."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get model dimensions
    model_size_conv = model.model_complexity_conv * 16
    model_size_lstm = model.model_complexity_lstm * 16
    
    print(f"Model dimensions: conv={model_size_conv}, lstm={model_size_lstm}")
    
    # Export just the step function using TorchScript
    class StepWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, acoustic_out, hidden_h, hidden_c, prev_output):
            hidden = (hidden_h, hidden_c)
            language_out, next_hidden = self.model.lm_model_step(acoustic_out, hidden, prev_output)
            return language_out, next_hidden[0], next_hidden[1]
    
    wrapper = StepWrapper(model)
    wrapper.eval()
    
    # Create example inputs
    acoustic_input = torch.randn(1, 1, model_size_conv)
    prev_output = torch.zeros(1, 1, 88).long()
    h0 = torch.zeros(2, 1, model_size_lstm)
    c0 = torch.zeros(2, 1, model_size_lstm)
    
    print(f"Input shapes:")
    print(f"  - Acoustic: {acoustic_input.shape}")
    print(f"  - Hidden h: {h0.shape}")
    print(f"  - Hidden c: {c0.shape}")
    print(f"  - Prev output: {prev_output.shape}")
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(wrapper, (acoustic_input, h0, c0, prev_output))
        
        # Save TorchScript
        script_path = output_path / "step_model.pt"
        traced_model.save(str(script_path))
        print(f"TorchScript saved to: {script_path}")
        
        # Get the graph representation
        graph = traced_model.graph
        print(f"Graph nodes: {len(list(graph.nodes()))}")
        
        # Save graph representation
        graph_path = output_path / "step_model_graph.txt"
        with open(graph_path, 'w') as f:
            f.write(str(graph))
        print(f"Graph representation saved to: {graph_path}")
        
        return traced_model, (acoustic_input, h0, c0, prev_output)
        
    except Exception as e:
        print(f"Error during tracing: {e}")
        return None, None


def export_using_torch_compile(model, output_dir: str = "exports"):
    """Try exporting using torch.compile if available."""
    
    if not hasattr(torch, 'compile'):
        print("torch.compile not available in this PyTorch version")
        return None, None
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    class StepWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, acoustic_out, hidden_h, hidden_c, prev_output):
            hidden = (hidden_h, hidden_c)
            language_out, next_hidden = self.model.lm_model_step(acoustic_out, hidden, prev_output)
            return language_out, next_hidden[0], next_hidden[1]
    
    wrapper = StepWrapper(model)
    wrapper.eval()
    
    # Get model dimensions
    model_size_conv = model.model_complexity_conv * 16
    model_size_lstm = model.model_complexity_lstm * 16
    
    # Create example inputs
    acoustic_input = torch.randn(1, 1, model_size_conv)
    prev_output = torch.zeros(1, 1, 88).long()
    h0 = torch.zeros(2, 1, model_size_lstm)
    c0 = torch.zeros(2, 1, model_size_lstm)
    
    try:
        print("Attempting torch.compile...")
        compiled_model = torch.compile(wrapper, backend="aot_eager")
        
        # Run the compiled model
        with torch.no_grad():
            output = compiled_model(acoustic_input, h0, c0, prev_output)
        
        print(f"Compiled model output shapes: {[o.shape for o in output]}")
        return compiled_model, (acoustic_input, h0, c0, prev_output)
        
    except Exception as e:
        print(f"Error during torch.compile: {e}")
        return None, None


def export_to_onnx_detailed(model, output_dir: str = "exports"):
    """Enhanced ONNX export with more detailed configuration."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    class StepWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, acoustic_out, hidden_h, hidden_c, prev_output):
            hidden = (hidden_h, hidden_c)
            language_out, next_hidden = self.model.lm_model_step(acoustic_out, hidden, prev_output)
            return language_out, next_hidden[0], next_hidden[1]
    
    wrapper = StepWrapper(model)
    wrapper.eval()
    
    # Get model dimensions
    model_size_conv = model.model_complexity_conv * 16
    model_size_lstm = model.model_complexity_lstm * 16
    
    # Create example inputs
    acoustic_input = torch.randn(1, 1, model_size_conv)
    prev_output = torch.zeros(1, 1, 88).long()
    h0 = torch.zeros(2, 1, model_size_lstm)
    c0 = torch.zeros(2, 1, model_size_lstm)
    
    onnx_path = output_path / "step_model_detailed.onnx"
    
    try:
        torch.onnx.export(
            wrapper,
            (acoustic_input, h0, c0, prev_output),
            str(onnx_path),
            export_params=True,
            opset_version=17,  # Use newer opset
            do_constant_folding=True,
            input_names=['acoustic_out', 'hidden_h', 'hidden_c', 'prev_output'],
            output_names=['language_out', 'next_hidden_h', 'next_hidden_c'],
            dynamic_axes={
                # Only make time dimension dynamic, keep batch size fixed at 1
                'acoustic_out': {1: 'time'},
                'prev_output': {1: 'time'},
                'language_out': {1: 'time'},
                # Remove batch dimension from hidden states to avoid LSTM warning
                # 'hidden_h': {1: 'batch'},  # Removed
                # 'hidden_c': {1: 'batch'},  # Removed
                # 'next_hidden_h': {1: 'batch'},  # Removed
                # 'next_hidden_c': {1: 'batch'},  # Removed
            },
            verbose=True
        )
        print(f"Detailed ONNX model saved to: {onnx_path}")
        print("✅ ONNX export configured to avoid LSTM batch size warnings")
        return str(onnx_path)
        
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        return None


def export_to_onnx_batch_1_fixed(model, output_dir: str = "exports"):
    """ONNX export with fixed batch size = 1 to eliminate all warnings."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    class StepWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, acoustic_out, hidden_h, hidden_c, prev_output):
            hidden = (hidden_h, hidden_c)
            language_out, next_hidden = self.model.lm_model_step(acoustic_out, hidden, prev_output)
            return language_out, next_hidden[0], next_hidden[1]
    
    wrapper = StepWrapper(model)
    wrapper.eval()
    
    # Get model dimensions
    model_size_conv = model.model_complexity_conv * 16
    model_size_lstm = model.model_complexity_lstm * 16
    
    # Create example inputs with fixed batch size = 1
    acoustic_input = torch.randn(1, 1, model_size_conv)
    prev_output = torch.zeros(1, 1, 88).long()
    h0 = torch.zeros(2, 1, model_size_lstm)
    c0 = torch.zeros(2, 1, model_size_lstm)
    
    onnx_path = output_path / "step_model_batch1_fixed.onnx"
    
    try:
        torch.onnx.export(
            wrapper,
            (acoustic_input, h0, c0, prev_output),
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['acoustic_out', 'hidden_h', 'hidden_c', 'prev_output'],
            output_names=['language_out', 'next_hidden_h', 'next_hidden_c'],
            # No dynamic_axes at all - everything is fixed batch size = 1
            dynamic_axes=None,
            verbose=True
        )
        print(f"Fixed batch=1 ONNX model saved to: {onnx_path}")
        print("✅ No dynamic batch dimensions - eliminates all LSTM warnings")
        return str(onnx_path)
        
    except Exception as e:
        print(f"Error during fixed batch ONNX export: {e}")
        return None


def convert_onnx_to_mlir_manual(onnx_path: str, output_dir: str = "exports"):
    """Manually convert ONNX to MLIR using available tools."""
    
    if not onnx_path or not os.path.exists(onnx_path):
        print("ONNX file not found, skipping MLIR conversion")
        return None
    
    output_path = Path(output_dir)
    mlir_path = output_path / "step_model_from_onnx.mlir"
    
    # Try using iree-import-onnx if available
    import subprocess
    import shutil
    
    # Check if iree-import-onnx is available
    if shutil.which("iree-import-onnx"):
        try:
            cmd = [
                "iree-import-onnx",
                onnx_path,
                "-o", str(mlir_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"MLIR from ONNX saved to: {mlir_path}")
                return str(mlir_path)
            else:
                print(f"iree-import-onnx failed: {result.stderr}")
        except Exception as e:
            print(f"Error running iree-import-onnx: {e}")
    else:
        print("iree-import-onnx not found in PATH")
    
    return None


def compile_mlir_to_vmfb(mlir_path: str, target_backend: str, output_dir: str = "exports"):
    """Compile MLIR to VMFB using iree-compile."""
    
    if not mlir_path or not os.path.exists(mlir_path):
        print("MLIR file not found, skipping VMFB compilation")
        return None
    
    import subprocess
    import shutil
    
    output_path = Path(output_dir)
    vmfb_path = output_path / f"step_model_{target_backend}.vmfb"
    
    # Check if iree-compile is available
    if not shutil.which("iree-compile"):
        print("iree-compile not found in PATH")
        return None
    
    try:
        # Set compilation options based on target
        cmd = ["iree-compile", mlir_path, "-o", str(vmfb_path)]
        
        if target_backend == 'vulkan':
            cmd.extend([
                "--iree-hal-target-backends=vulkan-spirv",
                "--iree-vulkan-target-triple=unknown-unknown-unknown"
            ])
        elif target_backend == 'metal':
            cmd.extend([
                "--iree-hal-target-backends=metal-spirv",
                "--iree-metal-target-platform=macos"
            ])
        else:  # cpu
            cmd.extend([
                "--iree-hal-target-backends=llvm-cpu",
                "--iree-llvmcpu-target-triple=x86_64-linux-gnu"
            ])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"VMFB for {target_backend} saved to: {vmfb_path}")
            return str(vmfb_path)
        else:
            print(f"iree-compile failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error during VMFB compilation: {e}")
        return None


def main():
    """Main function to export the model using fallback methods."""
    
    # Load the model
    model_path = "../model-180000.pt"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print("Loading model...")
    model = load_model(model_path)
    model.eval()
    
    # Create exports directory
    output_dir = ".exports"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Method 1: TorchScript Export ===")
    traced_model, inputs = simple_export_to_mlir(model, output_dir)
    
    print("\n=== Method 2: torch.compile (if available) ===")
    compiled_model, _ = export_using_torch_compile(model, output_dir)
    
    print("\n=== Method 3: Enhanced ONNX Export (with time-dynamic) ===")
    onnx_path = export_to_onnx_detailed(model, output_dir)
    
    print("\n=== Method 3b: Fixed Batch=1 ONNX Export (eliminates warnings) ===")
    onnx_fixed_path = export_to_onnx_batch_1_fixed(model, output_dir)
    
    print("\n=== Method 4: ONNX to MLIR Conversion ===")
    # Try the fixed batch version first, fallback to dynamic if needed
    mlir_path = convert_onnx_to_mlir_manual(onnx_fixed_path or onnx_path, output_dir)
    
    print("\n=== Method 5: MLIR to VMFB Compilation ===")
    if mlir_path:
        backends = ['cpu', 'vulkan', 'metal']
        for backend in backends:
            print(f"\nCompiling for {backend}...")
            vmfb_path = compile_mlir_to_vmfb(mlir_path, backend, output_dir)
    
    print("\n=== Export Complete ===")
    print("Generated files:")
    for file in Path(output_dir).glob("*"):
        file_size = file.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"  - {file.name} ({file_size:.2f} MB)")


if __name__ == "__main__":
    main()
