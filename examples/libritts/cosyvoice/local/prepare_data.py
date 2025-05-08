import argparse
import logging
import os
from tqdm import tqdm

logger = logging.getLogger()


def main():
    # Create destination directory if it doesn't exist
    if not os.path.exists(args.des_dir):
        os.makedirs(args.des_dir, exist_ok=True)

    # Determine speaker_id from the destination directory name
    # Example: if args.des_dir is 'data/myvoicespeaker_train', speaker_id will be 'myvoicespeaker_train'
    speaker_id = os.path.basename(os.path.normpath(args.des_dir))
    if not speaker_id:  # Fallback if des_dir is like '/' or empty after normpath
        speaker_id = "default_speaker"
        logger.warning(
            f"Could not determine a valid speaker ID from des_dir '{args.des_dir}'. Using '{speaker_id}'."
        )

    utt2wav, utt2text, utt2spk = {}, {}, {}
    # For a single speaker dataset, spk2utt will have one entry
    spk2utt = {speaker_id: []}

    input_filelist_path = args.input_filelist

    if not os.path.isfile(input_filelist_path):
        logger.error(f"Input filelist not found: {input_filelist_path}")
        return

    try:
        with open(input_filelist_path, 'r', encoding='utf-8') as f_filelist:
            # Read all lines to use with tqdm
            lines = f_filelist.readlines()
    except Exception as e:
        logger.error(f"Error reading input filelist {input_filelist_path}: {e}")
        return

    for line in tqdm(lines, desc=f"Processing filelist for speaker {speaker_id}"):
        line = line.strip()
        if not line:
            continue

        parts = line.split('|', 1)
        if len(parts) != 2:
            logger.warning(f"Skipping malformed line (expected 'wav_path|text'): {line}")
            continue
        
        wav_path_relative, text_content = parts[0].strip(), parts[1].strip()
        
        if not wav_path_relative:
            logger.warning(f"Skipping line with empty wav path: {line}")
            continue
        
        # Ensure the wav_path is absolute for robustness
        wav_path_absolute = os.path.abspath(wav_path_relative)

        if not os.path.isfile(wav_path_absolute):
            logger.warning(f"WAV file not found: '{wav_path_absolute}' (from line: '{wav_path_relative}'). Skipping entry.")
            continue
        
        # Extract base filename without extension, e.g., "0001" from "path/to/0001.wav"
        base_filename = os.path.splitext(os.path.basename(wav_path_absolute))[0]

        # Create utterance ID, e.g., "myvoicespeaker_train_0001"
        utt_id = f"{speaker_id}_{base_filename}"

        # Store mappings
        utt2wav[utt_id] = wav_path_absolute
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
            logger.warning("No utterances were processed. Please check your input filelist and WAV paths.")
        # The original script didn't print a success message, so we don't either.

    except IOError as e:
        logger.error(f"Error writing output files to {args.des_dir}: {e}")
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
    parser = argparse.ArgumentParser(description="Prepare data for CosyVoice single-speaker fine-tuning from a filelist.")
    parser.add_argument('--input_filelist',
                        type=str,
                        required=True,
                        help="Path to the input filelist. Each line: 'path/to/audio.wav|transcript'.")
    parser.add_argument('--des_dir',
                        type=str,
                        required=True,
                        help="Destination directory to save the output files (wav.scp, text, etc.).")
    args = parser.parse_args()
    main()
