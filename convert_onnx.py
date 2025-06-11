# export_to_onnx.py
import torch
from transcribe import load_model

class StepWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, acoustic_out, hidden_h, hidden_c, prev_output):
        hidden = (hidden_h, hidden_c)
        language_out, next_hidden = self.model.lm_model_step(acoustic_out, hidden, prev_output)
        return language_out, next_hidden[0], next_hidden[1]


def export_model(model, output_path: str):
    wrapper = StepWrapper(model)
    
    # Get the actual model dimensions from the loaded model
    model_size_conv = model.model_complexity_conv * 16
    model_size_lstm = model.model_complexity_lstm * 16
    
    # Create input tensors with correct dimensions
    acoustic_input = torch.randn(1, 1, model_size_conv)  # Use actual conv output size
    prev_output = torch.zeros(1, 1, 88).long()
    h0 = torch.zeros(2, 1, model_size_lstm)  # Use actual LSTM hidden size
    c0 = torch.zeros(2, 1, model_size_lstm)  # Use actual LSTM hidden size

    torch.onnx.export(
        wrapper,
        (acoustic_input, h0, c0, prev_output),
        output_path,
        export_params=True,
        input_names=['acoustic_out', 'hidden_h', 'hidden_c', 'prev_output'],
        output_names=['language_out', 'next_hidden_h', 'next_hidden_c'],
        dynamic_axes={
            'acoustic_out': {0: 'batch', 1: 'time'},
            'prev_output': {0: 'batch', 1: 'time'},
            'language_out': {0: 'batch', 1: 'time'},
        },
        opset_version=13
    )


if __name__ == "__main__":
    model = load_model('./model-180000.pt')
    model.eval()

    export_model(model, 'model.onnx')