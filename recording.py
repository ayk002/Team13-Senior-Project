import subprocess
import sys
import serial
import wave
import time
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting

def install_package(package_name):
    """Install a Python package using pip."""
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])

# Ensure numpy and matplotlib are installed
try:
    import numpy as np
except ImportError as e:
    print(f"{e.name} not found. Installing {e.name}...")
    install_package('numpy')
    import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"{e.name} not found. Installing {e.name}...")
    install_package('matplotlib')
    import matplotlib.pyplot as plt

# Define the serial port and baud rate
port = '/dev/tty.usbmodem138234301'  # Adjust this to match your port
baud_rate = 115200  # Should match the baud rate in your Arduino code

# Parameters for audio file
sample_rate = 44100  # Sample rate in Hz
sample_width = 2     # Sample width in bytes (2 bytes for 16-bit audio)
num_channels = 1     # Mono audio
record_duration = 10  # Duration to record in seconds

def log_message(message):
    """Helper function to log messages with timestamps."""
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def normalize_data(data, max_value, sample_width):
    """Normalize data to fit in the audio format."""
    max_amplitude = (2 ** (sample_width * 8 - 1)) - 1  # Max amplitude for the given sample width
    data_array = np.array(data, dtype=np.float32)
    normalized_data = np.int16(data_array * max_amplitude / max_value)
    return normalized_data.tobytes()

def plot_data(data_buffer):
    """Plot the audio data over time."""
    plt.figure(figsize=(10, 4))
    plt.plot(data_buffer)
    plt.title('Audio Data over Time')
    plt.xlabel('Sample Number')
    plt.ylabel('Analog Value')
    plt.grid()
    plt.show()

def plot_frequency_spectrum(data_buffer):
    """Plot the frequency spectrum of the audio data."""
    # Compute the FFT of the data
    fft_data = np.fft.fft(data_buffer)
    
    # Calculate the magnitude of the FFT (absolute value)
    fft_magnitude = np.abs(fft_data)
    
    # Frequency axis (in Hz)
    freqs = np.fft.fftfreq(len(data_buffer), 1 / sample_rate)
    
    # Only keep the positive half of the frequencies (real-valued signal)
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
    
    # Plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_magnitude)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

def record_audio():
    """Record audio from the serial port and save it to a WAV file."""
    start_time = time.time()
    end_time = start_time + record_duration

    all_data_buffer = []  # Collect all data

    try:
        log_message(f"Attempting to open serial port {port} with baud rate {baud_rate}.")
        with serial.Serial(port, baud_rate, timeout=1) as ser:
            log_message("Connected to serial port.")
            
            while time.time() < end_time:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line.isdigit():
                        data_value = int(line)
                        all_data_buffer.append(data_value)

    except serial.SerialException as e:
        log_message(f"Serial exception: {e}")
    except Exception as e:
        log_message(f"An unexpected error occurred: {e}")

    # Write all collected data to the WAV file
    if all_data_buffer:
        log_message("Writing collected data to output_raw.wav.")
        try:
            with wave.open('output_raw.wav', 'wb') as audio_file:
                log_message("Opened output_raw.wav for writing.")
                audio_file.setnchannels(num_channels)
                audio_file.setsampwidth(sample_width)
                audio_file.setframerate(sample_rate)

                normalized_data = normalize_data(all_data_buffer, 1023, sample_width)
                audio_file.writeframes(normalized_data)

        except Exception as e:
            log_message(f"Error writing to WAV file: {e}")

    # Plot the audio data
    plot_data(all_data_buffer)

    # Plot the frequency spectrum of the data
    plot_frequency_spectrum(all_data_buffer)

    log_message("Recording finished.")

if __name__ == "__main__":
    record_audio()