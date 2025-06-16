# Examples

This directory contains example scripts demonstrating different aspects of the online automatic music transcription system.

## Files

### `pyaudio_tutorial.py`
A tutorial demonstrating basic PyAudio functionality and microphone input handling. This example shows how to:
- Set up microphone streaming
- Process audio data in real-time
- Basic integration with the transcription model

### `pyplot_test.py`
A simple test script for matplotlib real-time plotting functionality. This demonstrates:
- Setting up matplotlib for real-time updates
- Interactive plotting with Qt5Agg backend
- Basic animation loop for data visualization

### `run_on_plt.py`
A matplotlib-based runner for the transcription system. This example shows:
- Real-time audio transcription with matplotlib visualization
- Integration of microphone streaming with the transcription model
- Live plotting of transcription results

## Usage

Make sure you have the required dependencies installed:
```bash
pip install -r ../requirements.txt
```

Note: Some examples may require additional system dependencies like ALSA or audio drivers, especially on Linux systems.

## Running Examples

From the main project directory:
```bash
python examples/pyplot_test.py
python examples/run_on_plt.py
python examples/pyaudio_tutorial.py
```

Make sure your microphone is properly configured and accessible before running the audio-related examples.
