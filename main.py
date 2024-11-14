import sys
import librosa
import numpy as np
import yt_dlp
import os


def download_video(url, output_path, progress_callback=None):
    def progress_hook(d):
        if d['status'] == 'downloading' and progress_callback:
            try:
                percent = d.get('_percent_str', '0%').replace('%', '')
                progress_callback(int(float(percent)))
            except ValueError:
                pass  # Ignore if we can't convert the percentage to a number

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_path, "downloads", "%(title)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "progress_hooks": [progress_hook],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            final_filename = os.path.splitext(filename)[0] + ".mp3"
            if os.path.exists(final_filename):
                return final_filename
            else:
                return f"Error: File not found after download: {final_filename}"
    except Exception as e:
        return f"Error: {str(e)}"
    
def main():
    # Check if the audio file path is provided
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_file_path>")
        sys.exit(1)

    # Get the audio file path from the command-line arguments
    audio_path = sys.argv[1]
    if "https:" in audio_path:
        audio_path = download_video(audio_path, "downloads")

    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Apply a low-pass filter to isolate bass frequencies
    def low_pass_filter(signal, sr, cutoff=200):
        fft = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(fft))
        cutoff_idx = np.abs(frequencies * sr) > cutoff
        fft[cutoff_idx] = 0
        filtered_signal = np.fft.ifft(fft)
        return filtered_signal.real

    y_filtered = low_pass_filter(y, sr)

    # Onset detection to find note beginnings
    onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr, hop_length=512)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512)

    # Convert frame indices to timestamps
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    # Estimate pitches at each onset
    pitches = []
    for onset in onset_frames:
        # Get a short window around the onset
        start = onset
        end = onset + 2  # Adjust the window size as needed
        y_slice = y_filtered[start * 512:end * 512]
        # Estimate pitch
        pitch, mag = librosa.piptrack(y=y_slice, sr=sr)
        # Find the highest magnitude bin
        index = mag.argmax()
        frequency = pitch.flatten()[index]
        # Handle cases where pitch detection fails
        if frequency == 0:
            pitches.append(None)
        else:
            # Convert frequency to MIDI note
            midi_note = librosa.hz_to_midi(frequency)
            pitches.append(midi_note)

    # Map MIDI notes to bass guitar strings and frets
    # Standard bass tuning: E1 (40), A1 (45), D2 (50), G2 (55)
    string_notes = [40, 45, 50, 55]  # MIDI note numbers for open strings
    tab_lines = ['G|', 'D|', 'A|', 'E|']

    # Initialize tab lines
    num_onsets = len(onset_times)
    tab_length = num_onsets * 3  # Adjust spacing as needed
    for i in range(len(tab_lines)):
        tab_lines[i] += '-' * tab_length

    # Place notes on tab lines
    for i, midi_note in enumerate(pitches):
        if midi_note is None:
            continue  # Skip if no pitch was detected
        fret = None
        string = None
        # Find the string and fret
        for s, open_note in enumerate(reversed(string_notes)):
            fret_candidate = int(round(midi_note - open_note))
            if 0 <= fret_candidate <= 24:  # Typical bass fret range
                fret = fret_candidate
                string = len(string_notes) - 1 - s
                break
        if fret is not None and string is not None:
            # Calculate the position in the tab line
            pos = i * 3 + 2  # Adjust spacing as needed
            fret_str = str(fret).ljust(2, '-')
            tab_lines[string] = tab_lines[string][:pos] + fret_str + tab_lines[string][pos+2:]

    # Print the tab
    print('\n'.join(tab_lines))

if __name__ == "__main__":
    main()
