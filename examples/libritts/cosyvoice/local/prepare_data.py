import argparse
import logging
import os
from tqdm import tqdm

logger = logging.getLogger()


def main():
    # Create destination directory if it doesn't exist
    if not os.path.exists(args.des_dir):
        os.makedirs(args.des_dir, exist_ok=True)

    # Determine speaker_id from the source directory name
    # Example: if args.src_dir is 'data/myvoice', speaker_id will be 'myvoice'
    speaker_id = os.path.basename(os.path.normpath(args.src_dir))
    if not speaker_id:  # Fallback if src_dir is like '/' or empty after normpath
        speaker_id = "default_speaker"
        logger.warning(
            f"Could not determine a valid speaker ID from src_dir '{args.src_dir}'. Using '{speaker_id}'."
        )

    utt2wav, utt2text, utt2spk = {}, {}, {}
    # For a single speaker dataset, spk2utt will have one entry
    spk2utt = {speaker_id: []}

    transcripts_file_path = os.path.join(args.src_dir, 'transcripts.txt')
    wav_files_dir = os.path.join(args.src_dir, 'wavs')

    if not os.path.isfile(transcripts_file_path):
        logger.error(f"Transcripts file not found: {transcripts_file_path}")
        return
    
    if not os.path.isdir(wav_files_dir):
        logger.error(f"WAVs directory not found: {wav_files_dir}")
        return

    try:
        with open(transcripts_file_path, 'r', encoding='utf-8') as f_transcripts:
            # Read all lines to use with tqdm
            lines = f_transcripts.readlines()
    except Exception as e:
        logger.error(f"Error reading transcripts file {transcripts_file_path}: {e}")
        return

    for line in tqdm(lines, desc=f"Processing transcripts for speaker {speaker_id}"):
        line = line.strip()
        if not line:
            continue

        parts = line.split('|', 1)
        if len(parts) != 2:
            logger.warning(f"Skipping malformed line (expected 'filename|text'): {line}")
            continue
        
        base_filename, text_content = parts[0].strip(), parts[1].strip()
        
        if not base_filename:
            logger.warning(f"Skipping line with empty base filename: {line}")
            continue

        # Create utterance ID, e.g., "myvoice_recording_001"
        utt_id = f"{speaker_id}_{base_filename}"
        # Create full path to the WAV file
        wav_path = os.path.join(wav_files_dir, f"{base_filename}.wav")

        if not os.path.isfile(wav_path):
            logger.warning(f"WAV file not found for '{base_filename}': {wav_path}. Skipping entry.")
            continue

        # Store mappings
        # Using absolute paths for wav.scp is generally more robust
        utt2wav[utt_id] = os.path.abspath(wav_path)
        utt2text[utt_id] = text_content
        utt2spk[utt_id] = speaker_id
        spk2utt[speaker_id].append(utt_id)

    # Write output files
    output_wav_scp = os.path.join(args.des_dir, 'wav.scp')
    output_text = os.path.join(args.des_dir, 'text')
    output_utt2spk = os.path.join(args.des_dir, 'utt2spk')
    output_spk2utt = os.path.join(args.des_dir, 'spk2utt')

    try:
        with open(output_wav_scp, 'w', encoding='utf-8') as f:
            for k, v in utt2wav.items():
                f.write(f'{k} {v}\n')
        
        with open(output_text, 'w', encoding='utf-8') as f:
            for k, v in utt2text.items():
                f.write(f'{k} {v}\n')

        with open(output_utt2spk, 'w', encoding='utf-8') as f:
            for k, v in utt2spk.items():
                f.write(f'{k} {v}\n')

        with open(output_spk2utt, 'w', encoding='utf-8') as f:
            # There's only one speaker, but this loop is general
            for spk_id_key, utt_list in spk2utt.items():
                f.write(f'{spk_id_key} {" ".join(utt_list)}\n')
        
        if not utt2wav:
            logger.warning("No utterances were processed. Please check your data and transcripts.txt.")
        # The original script didn't print a success message, so we don't either.

    except IOError as e:
        logger.error(f"Error writing output files to {args.des_dir}: {e}")
        return
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for CosyVoice single-speaker fine-tuning.")
    parser.add_argument('--src_dir',
                        type=str,
                        required=True,
                        help="Source directory containing 'transcripts.txt' and a 'wavs/' subdirectory.")
    parser.add_argument('--des_dir',
                        type=str,
                        required=True,
                        help="Destination directory to save the output files (wav.scp, text, etc.).")
    args = parser.parse_args()
    main()
