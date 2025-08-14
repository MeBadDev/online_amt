


import librosa
import numpy as np
from transcribe import load_model, OnlineTranscriber
import argparse
import time

def main(audio_file, model_file):
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    print(f"Loaded {audio_file}: {y.shape}, sr={sr}")

    model = load_model(model_file)
    # return_roll=False to get onsets/offsets
    transcriber = OnlineTranscriber(model, return_roll=False)

    frame_size = 512
    hop_size = 512
    n_frames = (len(y) - frame_size) // hop_size + 1

    current_time = 0.0
    for i in range(n_frames):
        frame = y[i*hop_size:i*hop_size+frame_size]
        if len(frame) < frame_size:
            break
        onsets, offsets = transcriber.inference(frame)
        t = (i * hop_size) / sr
        # Ensure onsets/offsets are always lists
        if isinstance(onsets, int):
            onsets = [onsets]
        if isinstance(offsets, int):
            offsets = [offsets]
        for pitch in onsets:
            # Add 21 to match MIDI note numbers
            pitch += 21
            print(f"Onset: time={t:.3f}s, midi_pitch={pitch}")
        for pitch in offsets:
            pitch += 21
            print(f"Offset: time={t:.3f}s, midi_pitch={pitch}")
        # Optional: simulate real-time by sleeping for frame duration
        # time.sleep(hop_size / sr)

    print("Done streaming audio file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # audio-test.mp3 contains 5 notes: C4, E4, G4, C5, C4 (assuming my piano is not out of tune lol)
    # The corresponding MIDI note should be: 60, 64, 67, 72, 60
    parser.add_argument('--audio_file', type=str, default='audio-test.mp3')
    parser.add_argument('--model_file', type=str, default='model-180000.pt')
    args = parser.parse_args()
    main(args.audio_file, args.model_file)
