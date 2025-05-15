from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import argparse

import sys
import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import os
from IPython.display import Audio

#  change line 48 to the path of the datafile that trains the model 
#  command line format: python rfcmodel.py "path to .wav file" "age" gender of patient
#  example use: python rfcmodel.py "path to .wav file" 51 male
#  example outputs: swallow 1 risk: low
#                   swallow 2 risk: high
#  PAS score classification: 1-2 low, 3-5 mid, 6-8 high

def main():
    '''
    parsing the commandline
    '''
    parser = argparse.ArgumentParser(description="Path of .wav file")
    parser.add_argument("audiopath", type=str, help="Path to the .wav file")
    parser.add_argument("age", type=int, help="Age of the Patient")
    parser.add_argument("gender", type=str, choices=["male", "female"], help="Gender of the patient")


    # Parse arguments
    args = parser.parse_args()

    # Store the argument in a variable
    audioPath = args.audiopath
    age = args.age
    gender = args.gender

    print(f"Audio file path: {audioPath}")
    print(f"Age: {age}")
    print(f"Gender: {gender}")

    male = 0 
    if gender == "male":
        male = 1

    audioPath = [audioPath]


    '''
    training the model 
    '''
    # df = pd.read_csv("/Users/jadeybabey/Desktop/SENIOR DESIGN/Machine Learning Model/10.16.24 Swallow Analysis_second_prototype_data.csv")
    # order of features: age	male	Number of swallows	Total Length (s):	Number of swallows	Swallow #	Length (s)	Swallow_attempts	Peak Amplitude	Average Amplitude	Amplitude^2	Average Amplitude^2	Area under curve	Average Area under curve	Average Frequency	Median Frequency	label	freq1	freq2	freq3	freq4	freq5
    # number of features: 22

    # Get the directory where this script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Build a path to the CSV in the same folder
    csv_path = os.path.join(BASE_DIR, "10.16.24 Swallow Analysis_second_prototype_data.csv")
    # Now read it
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['label'])  # All columns except the target
    y = df['label']

    train_ids =  [18, 11, 12, 36, 15, 7, 0, 35, 44, 1, 48, 34, 14, 26, 21, 27, 39, 3, 46, 8, 22, 9, 29, 19, 16, 23, 31, 33, 37, 42, 47, 24, 6, 32, 30, 40, 2, 20, 38]
    test_ids = [4, 10, 43, 5, 17, 25, 28, 45, 41, 13]

    train_df = df[df['ID'].isin(train_ids)]
    test_df = df[df['ID'].isin(test_ids)]

    X_train, y_train = train_df.drop(columns=['Unnamed: 0', 'ID','PAS', 'label']), train_df['label']
    X_test, y_test = test_df.drop(columns=['Unnamed: 0','ID','PAS','label']), test_df['label']


    rfc = RandomForestClassifier(n_estimators=200, random_state=42)
    rfc.fit(X_train, y_train)

    # y_pred = rfc.predict(X_test)


    '''
    takes in audio file and get data analysis
    from: andrewdyousef/Aspiration-Risk-Study SwallowAnalysis.py 
    '''
    # takes in audio path
    # To run a wav file, format as:
    # audioPath = ['filepath']
    # path:
    # audioPath = [r"C:\Users\Jade Chng\Desktop\Senior Design\AGM 6.25.2024 Applesauce 1.wav"]      #example

    #Load wav file as into array
    #y and sr are the standard used variables
    #y is array of amplitude, and sr is the sample-rate which is set to 22050 by default (so there is 22050 measurements per second)
    y, sr = librosa.load(audioPath[0])
    # When there are 2 or more wav files that must be concatonated do so like this
    if len(audioPath) > 0:
        for i in range(1, len(audioPath)):
            y_2, sr_2 = librosa.load(audioPath[i])
            y = np.concatenate((y, y_2))

    filename = audioPath[0][10:len(audioPath[0])-4]


    # This function sesctions the audiofile using the 'librosa.effects.split' function
    # It then uses modifies the result of this function to correctly segment 
    # The parameters can be modified so that the function will correctly segment the swallows
    # y: input amplitude array
    # sr: default sample rate
    # Parameters to modify:
    #   top_db (librosa parameter): The threshold (in decibels) below reference to consider as silence (usually 20 - 40)
    #   ref (librosa parameter): The reference amplitude. usually keep at 0.1
    #   gap_time: maximum amount of distance between audio segments to consider them the same swallow (usually 0.1 - 0.5)
    #   length_cutoff: The minimum amount of time to consider a segment a swallow (if less than segment is deleted) (usually 0-0.5)
    #   allowed_min_amplitude: The smallest max amplitude of a segment to consider it a swallow
    #   allowed_max_amplitude: The largest max amplitude of a segment to consider it a swallow
    #   merge: testing parameter, set to true only to return after librosa function but before modification (false by default)
    #   filename: filename of wav file
    #   saveparam: set to true to save parameters to 'Results\ParameterValues.xlsx', false by default
    segments = swallowsplit(y, sr, top_db=20, gap_time=.6, length_cutoff=0.2, allowed_min_amplitude=0.0, filename=filename)

    # print(segments/sr)

    GraphAmplitude(y, sr, segments)

    SeperateSwallows = [y[start:end] for start, end in segments]

    # output_matrix = [ [filename , '' , len(y)/sr , len(segments)] ]
    output_matrix = []

    avg_len = 0
    avg_num_swallows = 0
    avg_num_coughs = 0
    avg_peak_amp = 0
    avg_avg_amp = 0
    avg_amp_sqr = 0
    avg_amp_sqr_persec = 0
    avg_area_under = 0
    avg_area_under_persec = 0
    avg_avg_freq = 0
    avg_med_freq = 0
    for i in range(len(SeperateSwallows)):
        swallowNum = i+1
        length = round ( len(SeperateSwallows[i])/sr , 5 )

        individual_segments = swallowsplit(SeperateSwallows[i], sr, top_db=20, gap_time=0.6, length_cutoff=0, allowed_min_amplitude=0, allowed_max_amplitude=2)
        num_swallows = len(individual_segments)
        num_coughs = NumberCoughs(SeperateSwallows[i], sr)

        peak_amplitude = round ( SeperateSwallows[i].max() , 5 )
        avg_amplitude = round ( np.sum(abs(SeperateSwallows[i])) / len(SeperateSwallows[i]) , 5)
        amp_sqr = round ( np.sum(SeperateSwallows[i]**2) , 5 )
        area_under = np.trapz(np.abs(SeperateSwallows[i]))/sr

        average_peakfreq, median_peakfreq, max_five_frequencies = FrequencyFeatures(SeperateSwallows[i], sr)

        excel_row = [age , male , len(y)/sr , len(segments), swallowNum , length , num_swallows,  peak_amplitude , avg_amplitude , amp_sqr , amp_sqr/length , area_under , area_under/length ,     average_peakfreq, median_peakfreq, max_five_frequencies]
        output_matrix.append(excel_row)

        avg_len += length
        avg_num_swallows += num_swallows
        avg_num_coughs += num_coughs
        avg_peak_amp += peak_amplitude
        avg_avg_amp += avg_amplitude
        avg_amp_sqr += amp_sqr
        avg_amp_sqr_persec += amp_sqr/length
        avg_area_under += area_under
        avg_area_under_persec += area_under/length
        avg_avg_freq += average_peakfreq
        avg_med_freq += median_peakfreq

        #print("Peak Amplitude of Swallow ", swallowNum, " is: ", peak_amplitude)
        #print("Length of Swallow ", swallowNum, " is: ", len(SeperateSwallows[i])/sr)
        #energy = np.sum(SeperateSwallows[i]**2)
        #print("Energy of Swallow ", swallowNum, " is: ", energy)
        #print(ave_amplitude)
        #individual_segments = swallowsplit(SeperateSwallows[i], sr, top_db=18, gap_time=0, length_cutoff=0, allowed_min_amplitude=0)
        #GraphAmplitude(SeperateSwallows[i], sr, individual_segments)

    # num_s = len(segments)
    # avg_row = [ '' , '' , '' , '' , "Avg:" , avg_len/num_s , avg_num_swallows/num_s , avg_num_coughs/num_s , avg_peak_amp/num_s , avg_avg_amp/num_s , avg_amp_sqr/num_s , avg_amp_sqr_persec/num_s , avg_area_under/num_s , avg_area_under_persec/num_s , avg_avg_freq/num_s, avg_med_freq/num_s]
    # output_matrix.append(avg_row)
    # output_matrix.append([])

    # print(output_matrix)
        

    df_new = pd.DataFrame(output_matrix)
    df_expanded = df_new[15].apply(pd.Series)
    

    
    if len(df_new) == 0 or len(df_expanded.columns)<5:
        print("audio file too noisy or no swallows detected")
    
    else: 
        # Rename columns if necessary
        df_expanded.columns = [f"{15}_{i}" for i in range(1, 6)]

        # Concatenate with the original DataFrame (excluding column 15)
        df_new = pd.concat([df_new.drop(columns=[15]), df_expanded], axis=1)
        
        # replace any nan values with 0 
        df_new.fillna(0, inplace=True)
        # change feature names to str
        df_new.columns = df_new.columns.astype(str)
        df_new.to_numpy()
        # change column names
        df_new.columns = [
        "age", "male", "Number of swallows", "Total Length (s)", "Swallow #", "Length (s)", 
        "Swallow_attempts", "Peak Amplitude", "Average Amplitude", "Amplitude^2", 
        "Average Amplitude^2", "Area under curve", "Average Area under curve", 
        "Average Frequency", "Median Frequency", "freq1", "freq2", "freq3", "freq4", "freq5" ]

        df_new.to_csv("test.csv", index=False)

        '''
        do prediction from that audio file
        '''
        y_pred = rfc.predict(df_new)

        ## by per swallow
        for i in range(len(y_pred)): 
            if y_pred [i] == 0: 
                print("swallow", i+1, ": low risk")
            else: 
                print("swallow", i+1, ": risk of dysphagia")

        ## my maximum label
        if 1 in y_pred:
            print("Patient has risk of dysphagia.")
        else:
            print("Patient has no detected risk of dysphagia.")


    # print(y_pred)

def swallowsplit(y, sr, top_db=20, ref=.1, gap_time=.6, length_cutoff=0, allowed_min_amplitude=0, allowed_max_amplitude = 2, merge = True, filename='', saveparam=False):
    #Split array into different swallows
    segments = librosa.effects.split(y, top_db=top_db, ref=ref)

    #Attempting to limit loud audios
    filtered_segments = []
    for st, end in segments:
        segment = y[st:end]
        max_amplitude = max(abs(segment))
        if (allowed_min_amplitude <= max_amplitude <= allowed_max_amplitude):
            filtered_segments.append([st, end])
        #Trying to include cough in swallow. I think this is not necessary and might make it worse
        #elif (max_amplitude > allowed_max_amplitude) and (len(segment) < 0.4*sr):
        #    filtered_segments.append([st, end])
    segments = np.array(filtered_segments)
    if not merge: return segments

    i = 0
    #allowed time between peaks for 2 segments to be considered same swallow
    #gap_time
    while i < len(segments):
        [sti, endi] = segments[i]
        for j in range(len(segments)):
            if i == j: continue

            [stj, endj] = segments[j]
            if (stj > endi) and (stj - endi < gap_time*sr):
                if (j > i):
                    segments = np.delete(segments, j, 0)
                    segments = np.delete(segments, i, 0)
                else:
                    segments = np.delete(segments, i, 0)
                    segments = np.delete(segments, j, 0)
                segments = np.vstack((segments, [sti, endj]))
                i = -1
                break
            elif (sti > endj) and (sti - endj < gap_time*sr):
                if (j > i):
                    segments = np.delete(segments, j, 0)
                    segments = np.delete(segments, i, 0)
                else:
                    segments = np.delete(segments, i, 0)
                    segments = np.delete(segments, j, 0)
                segments = np.vstack((segments, [stj, endi]))
                i = -1
                break
            elif (sti > stj) and (endi < endj):
                segments = np.delete(segments, i, 0)
                i = -1
                break
            elif (stj > sti) and (endj < endi):
                segments = np.delete(segments, j, 0)
                i = -1
                break
            elif (sti < endj) and (endi > endj):
                segments = np.delete(segments, j, 0)
                segments = np.delete(segments, i, 0)
                segments = np.vstack((segments, [stj, endi]))
                i = -1
                break
            elif (stj < endi) and (endj > endi):
                segments = np.delete(segments, i, 0)
                segments = np.delete(segments, j, 0)
                segments = np.vstack((segments, [sti, endj]))
                i = -1
                break
        i = i + 1

    if len(segments) > 0: segments = segments[segments[:, 0].argsort()]

    i = 0
    while i < len(segments):
        segment = y[segments[i][0]:segments[i][1]]
        mean = np.mean(abs(segment))
        if segments[i][1] - segments[i][0] < length_cutoff*sr :
            segments = np.delete(segments, i, 0)
        #elif (mean > .0085) or (mean < .004):
        #    segments = np.delete(segments, i, 0)
        #elif (max(abs(segment)) < 0.05):
        #    segments = np.delete(segments, i, 0)
        else: i = i+1
    
    if saveparam:
        SaveParameter([ [filename, top_db, ref, gap_time, length_cutoff, allowed_min_amplitude, allowed_max_amplitude, merge, False, segments/sr] ])

    return segments

def SaveParameter(ParameterMatrix):
    df_new = pd.DataFrame(ParameterMatrix)
    existing_excel_path = './Results/ParameterValues.xlsx'
    df_existing = pd.read_excel(existing_excel_path)
    df_combined = pd.concat( [df_existing , df_new] , ignore_index=True)
    df_combined.to_excel(existing_excel_path, index=False)

def NumberCoughs(y, sr):
    segments = swallowsplit(y, sr, top_db=18, gap_time=0, length_cutoff=0, allowed_min_amplitude=0)
    num = 0
    for st, end in segments:
        segment = y[st:end]
        max_amplitude = np.max(np.abs(segment))
        if (max_amplitude > 1) and (len(segment) < .5*sr):
            num += 1
    return num
    
def LengthofSegments(segments, sr):
    swallowlengths = []
    for i in range(len(segments)):
        swallowlengths.append(segments[i][1] - segments[i][0])

    for i in range(len(swallowlengths)):
        print("Swallow ", i, " has length: ", swallowlengths[i] / sr, " seconds")

def GraphAmplitude(y, sr, segments=None): 
    #Plot Amplitude vs Time
    plt.figure(figsize=(10, 4))                         #create figure of size 10x4
    librosa.display.waveshow(y, sr=sr, color="blue")    #create graph of y (amplitude) by sr (samples per second, aka time)
    plt.title('Time vs Amplitude')                      #title
    plt.xlabel('Time (s)')                              #x-axis label
    plt.ylabel('Amplitude')                             #y-axis label
    plt.xticks(np.arange(0, y.size/sr, 10))
    if not segments is None:
        for col in segments:
            for value in col:
                plt.axvline(x=value/sr, color='red')

    plt.show()                                          #display graph

def GraphSpectrogram(y, sr):
    #Creates a short-term forier transform from our amplitude over time array
    #Basically means each chunk of amplitude to see what frequency is present over time
    #This creates a 2d array with rows representing frequency and columns representing moments of time
    #Hop length is how much data will be represented in pixel of graph
    # - ie a smaller hop_length results in more detail
    fourier_transform = librosa.stft(y, hop_length=512, window='hamming')

    #Because fourier transform results in complex numbers, seperate into S (magnitude) and phase (imaginary component)
    #We could also do np.abs(fourier_transform) which would just result in S, but this stores the phase in case it can be useful later
    S, phase = librosa.magphase(fourier_transform)

    #Converts S to a form which uses decibels and sets relative to max
    #This is a 2d array which has frequency as the rows, time as the columns, and amplitude at each point
    S_db = librosa.amplitude_to_db(S, ref=np.max)


    plt.figure(figsize=(40, 4))                                                         # Create figure of size 40x4
    spectrogram = librosa.display.specshow(S_db, sr=sr, y_axis='log', x_axis='time')    # display spectrogram
    plt.title('Spectrogram')                                                            # set title
    plt.xlabel('Time')                                                                  # set x-axis
    plt.ylabel('Frequency (Hz)')                                                        # set y-axis
    plt.colorbar(spectrogram, format="%+2.0f dB").set_label('Amplitude')                # create legend for colors
    plt.show()                                                                          # display graph
    #NOTE
    # the legend shows the highest amplitude as 0, and all others in reference to this
    # this means 0 dB (white) is the max amplitude and black (which is negative) is the lowest

def FrequencyFeatures(y, sr):
    fourier_transform = librosa.stft(y, hop_length=512, window='hamming')
    S, phase = librosa.magphase(fourier_transform)
    S_db = librosa.amplitude_to_db(S, ref=np.max)                                                                       # display graph

    peak_frequencies = []
    fft_frequencies = librosa.fft_frequencies(sr=sr, n_fft=S_db.shape[0])

    for frame in S_db.T:                                                                #gets max freq per frame
        peak_freq_idx = np.argmax(frame)
        peak_freq = fft_frequencies[peak_freq_idx]
        peak_frequencies.append(peak_freq)
        
    average_peakfreq = np.mean(peak_frequencies)
    median_peakfreq = np.median(peak_frequencies)

    #print(f"Average Frequency:", average_peakfreq)
    #print(f"Median Frequency:", median_peakfreq)
    #print("Peak Frequencies:", peak_frequencies)
    unique_peak_frequencies = set(peak_frequencies)                                     #gets unique freq of list
    sorted_peak_frequencies = sorted(unique_peak_frequencies, reverse=True)             
    max_five_frequencies = sorted_peak_frequencies[:5]
    #print(f"Top 5 Peak Frequencies:", max_five_frequencies)

    return (average_peakfreq, median_peakfreq, max_five_frequencies)
   

def RemoveVoiceTest(y, sr):
    output_file_path = "testOutput.wav"

    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                margin_i * (S_full - S_filter),
                                power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                margin_v * S_filter,
                                power=power)

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    D_foreground = S_foreground * phase
    y_foreground = librosa.istft(D_foreground)
    sf.write(output_file_path, y_foreground, samplerate=sr, subtype='PCM_24')



if __name__ == "__main__":
    main()
