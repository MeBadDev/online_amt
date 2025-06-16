import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcribe import load_model, OnlineTranscriber

import matplotlib
matplotlib.use('Qt5Agg')
import pyaudio
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from mic_stream import MicrophoneStream

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "output.wav"


def main(args):
    stream = MicrophoneStream(RATE, CHUNK, CHANNELS)
    model = load_model(args.model_file)
    transcriber = OnlineTranscriber(model, return_roll=False)

    piano_roll = np.zeros((88, 32))
    piano_roll[30, 0] = 1
    entire_frames = []
    plt.ion()
    fig, ax = plt.subplots()

    x = np.arange(0, 2* CHUNK,2)
    plt.show(block=False)
    img = ax.imshow(piano_roll)
    ax_background = fig.canvas.copy_from_bbox(ax.bbox)
    ax.invert_yaxis()
    fig.canvas.draw()
    ONSETS = []

    with MicrophoneStream(RATE, CHUNK, CHANNELS) as stream:
        # 마이크 데이터 핸들을 가져옴 
        audio_generator = stream.generator()
        print("* recording")        
        for i in range(5000):
            data = stream._buff.get()
            time_a = time.time()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768
            if CHANNELS == 2:
                decoded = decoded.reshape(CHANNELS, -1)
                decoded = np.mean(decoded, axis=0)
            # frame_output = transcriber.inference(decoded)
            onset, offset = transcriber.inference(decoded)
            time_b = time.time()
            for pitch in onset:
                note_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][pitch % 12]
                octave = (pitch + 21) // 12
                print(f"Note ON: {note_name}{octave} (MIDI {pitch + 21})")
            for pitch in offset:
                note_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][pitch % 12]
                octave = (pitch + 21) // 12
                print(f"Note OFF: {note_name}{octave} (MIDI {pitch + 21})")
            if onset:
                print(f"Active notes: {onset}")
        stream.closed = True
    print("* done recording")


    # librosa.output.write_wav('lib_out.wav', np.concatenate(entire_frames), sr=44100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='/Users/jeongdasaem/Documents/model_weights/model-180000.pt')
    parser.add_argument('--rep_type', default='base')
    parser.add_argument('--n_class', default=5, type=int)
    parser.add_argument('--ac_model_type', default='simple_conv', type=str)
    parser.add_argument('--lm_model_type', default='lstm', type=str)
    parser.add_argument('--context_len', default=1, type=int)
    parser.add_argument('--no_recursive', action='store_true')
    args = parser.parse_args()

    main(args)