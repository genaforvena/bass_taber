import argparse
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
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Generate bass tabs from an audio file.')
    parser.add_argument('audio_file', help='Path to the audio file')
    parser.add_argument('-w', '--width', type=int, default=200, help='Maximum width of the output lines (default: 200)')
    args = parser.parse_args()

    audio_path = args.audio_file
    max_width = args.width

    if "https" in audio_path:
        audio_path = download_video(audio_path, "downloads")
    # Load the audio file
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except FileNotFoundError:
        print(f"Error: File '{audio_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)

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
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)

    # Convert frame indices to timestamps
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Estimate pitches at each onset
    pitches = []
    for onset in onset_frames:
        # Get a short window around the onset
        start = onset
        end = onset + 2  # Adjust the window size as needed
        y_slice = y_filtered[int(start * hop_length):int(end * hop_length)]
        # Estimate pitch
        pitch, mag = librosa.piptrack(y=y_slice, sr=sr)
        # Find the highest magnitude bin
        mag = mag.flatten()
        pitch = pitch.flatten()
        index = mag.argmax()
        frequency = pitch[index]
        # Handle cases where pitch detection fails
        if frequency == 0 or np.isnan(frequency):
            pitches.append(None)
        else:
            # Convert frequency to MIDI note
            midi_note = librosa.hz_to_midi(frequency)
            pitches.append(midi_note)

    # Map MIDI notes to bass guitar strings and frets
    # Standard bass tuning: E1 (40), A1 (45), D2 (50), G2 (55)
    string_notes = [40, 45, 50, 55]  # MIDI note numbers for open strings

    # Initialize tab characters for each string
    tab_chars = ['' for _ in range(4)]
    spacing = 3  # Spacing between notes

    # Build the tab characters
    for i, midi_note in enumerate(pitches):
        # Add spacing for each string
        for s in range(4):
            tab_chars[s] += '-' * spacing

        if midi_note is None:
            continue  # Skip if no pitch was detected

        fret = None
        string = None
        # Find the appropriate string and fret
        for s, open_note in enumerate(reversed(string_notes)):
            fret_candidate = int(round(midi_note - open_note))
            if 0 <= fret_candidate <= 24:  # Typical bass fret range
                fret = fret_candidate
                string = len(string_notes) - 1 - s
                break

        if fret is not None and string is not None:
            # Replace dashes with fret number
            fret_str = str(fret).ljust(spacing, '-')
            pos = i * spacing
            tab_chars[string] = tab_chars[string][:pos] + fret_str + tab_chars[string][pos + spacing:]

    # Split the tab lines according to the maximum width
    output_lines = []
    total_length = len(tab_chars[0])

    for start in range(0, total_length, max_width):
        end = start + max_width
        # Build tab lines for the current segment
        segment_lines = ['G|', 'D|', 'A|', 'E|']
        for s in range(4):
            line_segment = tab_chars[s][start:end]
            segment_lines[s] += line_segment
        # Add the segment to the output
        output_lines.extend(segment_lines)
        output_lines.append('')  # Add an empty line between segments

    # Prepare the output content
    output_content = '\n'.join(output_lines)

    # Determine the output file path
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_file = f"{base_name}.txt"

    # Write the tab to the output file
    try:
        with open(output_file, 'w') as f:
            f.write(output_content)
        print(f"Bass tabs have been written to '{output_file}'.")
    except Exception as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)        
        
if __name__ == "__main__":
    main()
