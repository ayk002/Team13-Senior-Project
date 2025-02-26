import wave
import numpy as np
from scipy.signal import stft, istft, butter, filtfilt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Parameters for processing
noise_duration = 1                # Duration of the noise segment in seconds
start_cut_duration = 0.5          # Cut the first 0.5 seconds from the audio
amplification_factor = 50         # Amplification factor after noise reduction
output_file_name = "output_SpectralGating_FrequencyBased_BandpassFiltered.wav"

# Define frequency ranges for low, mid, and high bands (in Hz)
LOW_FREQUENCY_BAND = (0, 500)      # Low frequencies: 0 Hz to 500 Hz (for hum, rumble, etc.)
MID_FREQUENCY_BAND = (700, 1200)   # Mid frequencies: 500 Hz to 1200 Hz
HIGH_FREQUENCY_BAND = (1200, 10000) # High frequencies: 1200 Hz to 5000 Hz (for hiss, etc.)

# Threshold factors for each frequency band
LOW_BAND_THRESHOLD_FACTOR = .45    # Stronger gating for low frequencies
MID_BAND_THRESHOLD_FACTOR = .45    # Moderate gating for mid frequencies
HIGH_BAND_THRESHOLD_FACTOR = .45   # Lighter gating for high frequencies

# Bandpass filter parameters
LOW_PASS = 300  # Low cut-off frequency for bandpass filter
HIGH_PASS = 1500  # High cut-off frequency for bandpass filter

def bandpass_filter(data, sample_rate, lowcut, highcut, order=5):
    """Apply a bandpass filter to the audio signal."""
    # Design bandpass filter
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter to the data using filtfilt (zero-phase filtering)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Apply frequency-dependent thresholds with a more conservative adjustment at low energy
def apply_spectral_gating(data, sample_rate, noise_segment, threshold_factors, smoothing_factor=0.000001):
    """Apply spectral gating with frequency-dependent thresholds to reduce noise."""
    f, t, Zxx = stft(data, fs=sample_rate, nperseg=256)
    
    _, _, Zxx_noise = stft(noise_segment, fs=sample_rate, nperseg=256)
    noise_magnitude = np.abs(Zxx_noise).mean(axis=1)
    
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    gated_magnitude = np.copy(magnitude)
    
    # Apply frequency-dependent thresholds with adjustments based on signal magnitude
    for i, freq in enumerate(f):
        if LOW_FREQUENCY_BAND[0] <= freq <= LOW_FREQUENCY_BAND[1]:
            gating_threshold = threshold_factors[0] * noise_magnitude[i]
        elif MID_FREQUENCY_BAND[0] <= freq <= MID_FREQUENCY_BAND[1]:
            gating_threshold = threshold_factors[1] * noise_magnitude[i]
        elif HIGH_FREQUENCY_BAND[0] <= freq <= HIGH_FREQUENCY_BAND[1]:
            gating_threshold = threshold_factors[2] * noise_magnitude[i]
        else:
            gating_threshold = threshold_factors[2] * noise_magnitude[i]
        
        # Adjust the gating threshold for low-energy regions
        gating_threshold = np.maximum(gating_threshold, 0.1)  # Avoid over-gating
        
        # Soft gating function with smoothing
        gated_magnitude[i, :] = np.maximum(magnitude[i, :] - gating_threshold, 0)
        gated_magnitude[i, :] = gated_magnitude[i, :] * np.exp(-magnitude[i, :] * smoothing_factor)
    
    gated_Zxx = gated_magnitude * np.exp(1j * phase)
    _, denoised_data = istft(gated_Zxx, fs=sample_rate)
    
    return denoised_data


def clip_audio(data):
    """Clip audio data to avoid exceeding the maximum amplitude."""
    max_amplitude = (2 ** (16 - 1)) - 1  # For 16-bit audio
    return np.clip(data, -max_amplitude, max_amplitude)

def process_audio(input_file):
    """Process the input WAV file and save it to the output file."""
    with wave.open(input_file, 'rb') as audio_file:
        params = audio_file.getparams()
        sample_rate = params.framerate
        num_frames = params.nframes
        audio_data = audio_file.readframes(num_frames)

    # Convert byte data to numpy array
    data_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Cut the first start_cut_duration seconds
    start_sample_index = int(start_cut_duration * sample_rate)
    data_array = data_array[start_sample_index:]
    
    # Extract noise segment for spectral gating
    noise_sample_count = int(noise_duration * sample_rate)
    if len(data_array) > noise_sample_count:
        noise_segment = data_array[:noise_sample_count]
        signal_data = data_array[noise_sample_count:]
    else:
        noise_segment = data_array
        signal_data = np.zeros_like(data_array)
    
    # Apply bandpass filter to isolate relevant frequencies
    filtered_data = bandpass_filter(signal_data, sample_rate, LOW_PASS, HIGH_PASS)
    
    # Frequency-dependent threshold factors
    threshold_factors = [LOW_BAND_THRESHOLD_FACTOR, MID_BAND_THRESHOLD_FACTOR, HIGH_BAND_THRESHOLD_FACTOR]
    
    # Apply spectral gating
    denoised_data = apply_spectral_gating(filtered_data, sample_rate, noise_segment, threshold_factors)
    
    # Amplify and clip the processed signal
    amplified_data = denoised_data * amplification_factor
    clipped_data = clip_audio(amplified_data)

    # Save processed data to a new WAV file
    with wave.open(output_file_name, 'wb') as output_audio_file:
        output_audio_file.setnchannels(1)  # Mono audio
        output_audio_file.setsampwidth(2)  # 16-bit audio
        output_audio_file.setframerate(sample_rate)
        output_audio_file.writeframes(clipped_data.astype(np.int16).tobytes())

if __name__ == "__main__":
    # Hide the root window
    Tk().withdraw()

    # Open file dialog to select the input file
    input_file = askopenfilename(title="Select the input WAV file", filetypes=[("WAV files", "*.wav")])
    if not input_file:
        print("No input file selected.")
        exit()

    process_audio(input_file)