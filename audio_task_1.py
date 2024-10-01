import sys
import librosa
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfilt

def low_pass_filter(audio_data, sr, cutoff=4000):

    sos = butter(10, cutoff, btype='low', fs=sr, output='sos')
    filtered_audio = sosfilt(sos, audio_data)
    
    return filtered_audio

def detect_notes(y, sr, min_duration=0.1):

    y = librosa.util.normalize(y)

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    pitches, magnitudes = librosa.core.piptrack(y=y_harmonic, sr=sr)

    pitches = median_filter(pitches, size=(1, 9))
    magnitudes = median_filter(magnitudes, size=(1, 9))

    note_pitches = []

    for onset in onset_frames:
        
        start = max(0, onset - 2)
        end = min(pitches.shape[1], onset + 2)

        pitch_window = pitches[:, start:end]
        magnitude_window = magnitudes[:, start:end]

        
        avg_pitch = np.mean(pitch_window, axis=1)
        avg_magnitude = np.mean(magnitude_window, axis=1)

        max_idx = avg_magnitude.argmax()
        note = avg_pitch[max_idx]
        if note > 0:
            note_pitches.append(note)

    
    note_names = librosa.hz_to_note(note_pitches)

    
    transcription = [(note_names[i], round(onset_times[i], 3), round(onset_times[i + 1] if i + 1 < len(onset_times) else y.shape[0] / sr, 3))
                     for i in range(len(note_names))]

    
    filtered_transcription = [note for note in transcription if note[2] - note[1] >= min_duration]

    return filtered_transcription


def main(audio_file_path):

    y, sr = librosa.load(audio_file_path, sr=None)
    
    y_filtered = low_pass_filter(y, sr)
    transcription = detect_notes(y_filtered, sr)

    print(transcription)

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python audio_task_1.py <path_to_audio_file>")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    main(audio_file_path)
