import sys
import os

import torch
#Set up the path to include the parent directory
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from transcribe import load_model

MODEL_PATH = 'model-180000.pt'

def export_model_to_onnx(model, output_path='model.onnx'):
    dummy_mel = torch.randn(1, 1, 229)
    dummy_gt_label = torch.zeros(0)  # Inference mode

    try:
        model.eval()

        torch.onnx.export(
            model,
            (dummy_mel, dummy_gt_label),
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['mel', 'gt_label'],
            output_names=['output'],
            dynamic_axes={
                'mel': {0: 'batch_size'},
                'gt_label': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )

        import onnx
        model_proto = onnx.load(output_path)
        onnx.checker.check_model(model_proto)
        print("ONNX export successful and model is valid!")
        return True

    except Exception as e:
        print(f"Error during ONNX export: {str(e)}")
        import traceback
        traceback.print_exc()
        return False



if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    export_model_to_onnx(model, 'model.onnx')
    print("Model exported to ONNX format.")