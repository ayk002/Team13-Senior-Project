import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import serial
import simpleaudio as sa  # For playing WAV files
import wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Piezo Mic Recorder")
        self.geometry("600x500")

        # Recording state and process handle
        self.recording = False
        self.recording_process = None

        # Start/Stop Recording Button
        self.record_btn = ttk.Button(self, text="Start Recording", command=self.toggle_recording)
        self.record_btn.grid(row=0, column=0, pady=10, padx=20)

        # Play Audio Button
        self.play_btn = ttk.Button(self, text="Play Audio", command=self.play_audio)
        self.play_btn.grid(row=1, column=0, pady=10)

        # Matplotlib Figure for Waveform Display
        self.fig, self.ax = plt.subplots(figsize=(5, 2))
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().grid(row=2, column=0, pady=10)

        # ML Interpretation Toggle
        self.ml_enabled = tk.BooleanVar()
        self.ml_toggle_btn = ttk.Checkbutton(self, text="Enable ML Interpretation", variable=self.ml_enabled)
        self.ml_toggle_btn.grid(row=3, column=0, pady=10)

        # Ensure a clean exit
        self.protocol("WM_DELETE_WINDOW", self.safe_exit)

    def toggle_recording(self):
        """
        Toggle recording:
        - When starting, launch recording.py using subprocess.
        - When stopping, terminate the external process (if still running).
        """
        if not self.recording:
            self.recording = True
            self.record_btn.config(text="Stop Recording")
            try:
                # Launch recording.py using the same Python interpreter as app.py
                self.recording_process = subprocess.Popen([sys.executable, "recording.py"])
                print("Recording started via external script.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start recording script:\n{e}")
                self.recording = False
                self.record_btn.config(text="Start Recording")
        else:
            self.recording = False
            self.record_btn.config(text="Start Recording")
            # Terminate the recording process if it's still running
            if self.recording_process is not None:
                self.recording_process.terminate()
                self.recording_process = None
                print("Recording process terminated.")

    def play_audio(self):
        """Play the recorded audio file."""
        try:
            wave_obj = sa.WaveObject.from_wave_file("output_raw.wav")
            wave_obj.play()
        except Exception as e:
            print(f"Error playing audio: {e}")
            messagebox.showerror("Error", "Could not play audio.")

    def plot_waveform(self, file_path="output_raw.wav"):
        """Load and display the waveform from a .wav file."""
        try:
            with wave.open(file_path, "rb") as wav_file:
                frames = wav_file.readframes(-1)
                audio_signal = np.frombuffer(frames, dtype=np.int16)
                self.ax.clear()
                self.ax.plot(audio_signal, color="blue")
                self.ax.set_title("Waveform")
                self.canvas.draw()
        except Exception as e:
            print(f"Error plotting waveform: {e}")
            messagebox.showerror("Error", "Could not load waveform.")

    def safe_exit(self):
        """Cleanly exit the application."""
        print("Closing application...")
        if self.recording_process is not None:
            self.recording_process.terminate()
        self.destroy()
        self.quit()

if __name__ == "__main__":
    app = AudioGUI()
    app.mainloop()
