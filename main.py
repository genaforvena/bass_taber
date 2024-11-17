import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import librosa
from spleeter.separator import Separator
import numpy as np
import crepe
import sys
import yt_dlp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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


def isolate_bass(input_file):
    separator = Separator('spleeter:4stems')
    output_dir = 'output'
    separator.separate_to_file(input_file, output_dir)
    # Return the path to the isolated bass file
    bass_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0], 'bass.wav')
    return bass_file

def detect_pitches(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    graph = tf.Graph()
    with graph.as_default():
        _, frequency, confidence, _ = crepe.predict(audio, sr=sr, viterbi=True)
    return frequency, confidence


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

    bass_file = isolate_bass(audio_path)
    frequency, confidence = detect_pitches(bass_file)

    # Convert frequency to MIDI notes
    def frequency_to_midi(frequency):
        # Avoid log of zero or negative frequencies
        with np.errstate(divide='ignore', invalid='ignore'):
            midi_notes = 69 + 12 * np.log2(frequency / 440.0)
        # Replace invalid values with NaN
        midi_notes = np.where(np.isfinite(midi_notes), midi_notes, np.nan)
        return midi_notes

    midi_notes = frequency_to_midi(frequency)

    print("Detected MIDI Notes:", midi_notes)

    # Map MIDI notes to bass guitar strings and frets
    # Standard bass tuning: E1 (40), A1 (45), D2 (50), G2 (55)
    string_notes = [40, 45, 50, 55]  # MIDI note numbers for open strings

    # Initialize tab characters for each string
    tab_chars = ['' for _ in range(4)]
    spacing = 3  # Spacing between notes

    # Build the tab characters
    for i, midi_note in enumerate(midi_notes):
        # Add spacing for each string
        for s in range(4):
            tab_chars[s] += '-' * spacing

        # Skip if no pitch was detected
        if np.isnan(midi_note):
            continue

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
