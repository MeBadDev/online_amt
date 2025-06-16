import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, jsonify
import pyaudio
from transcribe import load_model, OnlineTranscriber
from mic_stream import MicrophoneStream
import numpy as np
from threading import Thread
import queue
import time
from collections import deque

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
print('http://127.0.0.1:5000/')
app = Flask(__name__)
global Q, active_notes, note_history
Q = queue.Queue()
active_notes = set()  # Currently playing notes
note_history = deque(maxlen=100)  # Recent note events for visualization



@app.route('/')
def home():
    # args = Args()
    # model = load_model(args)
    model = load_model('model-180000.pt')
    global Q
    t1 = Thread(target=get_buffer_and_transcribe, name=get_buffer_and_transcribe, args=(model, Q))
    t1.start()
    return render_template('home.html')

@app.route('/_amt', methods= ['GET', 'POST'])
def amt():
    global Q
    onsets = []
    offsets = []
    while Q.qsize() > 0:
        rst = Q.get()
        onsets += rst[0]
        offsets += rst[1]
    return jsonify(on=onsets, off=offsets)

@app.route('/_notes', methods=['GET'])
def get_notes():
    global active_notes, note_history
    return jsonify(
        active_notes=list(active_notes),
        note_history=list(note_history)[-20:]  # Last 20 events
    )

def get_buffer_and_transcribe(model, q):
    global active_notes, note_history
    
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels']
    RATE = 16000

    print("Console Note Visualizer: Notes will be displayed here")
    print("=" * 60)

    def note_to_name(note_num):
        """Convert MIDI note number to note name"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (note_num + 21) // 12
        note = notes[(note_num + 21) % 12]
        return f"{note}{octave}"

    def print_active_notes():
        """Print currently active notes in a visual format"""
        if active_notes:
            note_names = [note_to_name(note) for note in sorted(active_notes)]
            print(f"\rActive Notes: {' | '.join(note_names):<50}", end='', flush=True)
        else:
            print(f"\rActive Notes: {'(silent)':<50}", end='', flush=True)

    stream = MicrophoneStream(RATE, CHUNK, CHANNELS)
    transcriber = OnlineTranscriber(model, return_roll=False)
    with MicrophoneStream(RATE, CHUNK, CHANNELS) as stream:
        audio_generator = stream.generator()
        print("* recording")
        on_pitch = []
        last_print_time = time.time()
        
        while True:
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768
            if CHANNELS > 1:
                decoded = decoded.reshape(-1, CHANNELS)
                decoded = np.mean(decoded, axis=1)
            frame_output = transcriber.inference(decoded)
            on_pitch += frame_output[0]
            
            # Handle note onsets
            for pitch in frame_output[0]:
                active_notes.add(pitch)
                note_history.append({
                    'type': 'onset',
                    'note': pitch,
                    'note_name': note_to_name(pitch),
                    'time': time.time()
                })
            
            # Handle note offsets
            for pitch in frame_output[1]:
                active_notes.discard(pitch)
                pitch_count = on_pitch.count(pitch)
                note_history.append({
                    'type': 'offset',
                    'note': pitch,
                    'note_name': note_to_name(pitch),
                    'time': time.time()
                })
            
            on_pitch = [x for x in on_pitch if x not in frame_output[1]]
            
            # Print active notes every 100ms to avoid too much console spam
            current_time = time.time()
            if current_time - last_print_time > 0.1:
                print_active_notes()
                last_print_time = current_time
            
            q.put(frame_output)
        stream.closed = True
    print("\n* done recording")

if __name__ == '__main__':
    # for i in range(0, p.get_device_count()):
    #     print(i, p.get_device_info_by_index(i)['name'])

    app.run(debug=True)
