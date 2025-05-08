#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import os
import json
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch

# Global variables for worker processes, to be initialized by init_worker
# These will hold the data dictionaries needed by the 'job' function.
g_utt2wav = None
g_utt2text = None
g_utt2spk = None
g_utt2embedding = None
g_spk2embedding = None
g_utt2speech_token = None

def init_worker(utt2wav_data, utt2text_data, utt2spk_data, utt2embedding_data, spk2embedding_data, utt2speech_token_data):
    """
    Initializes global variables for each worker process.
    This function is called by multiprocessing.Pool when each worker process starts.
    """
    global g_utt2wav, g_utt2text, g_utt2spk, g_utt2embedding, g_spk2embedding, g_utt2speech_token
    g_utt2wav = utt2wav_data
    g_utt2text = utt2text_data
    g_utt2spk = utt2spk_data
    g_utt2embedding = utt2embedding_data
    g_spk2embedding = spk2embedding_data
    g_utt2speech_token = utt2speech_token_data


def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()
    data_list = []
    for utt in tqdm(utt_list):
        data = open(g_utt2wav[utt], 'rb').read()  # Use global g_utt2wav
        data_list.append(data)
    wav_list = [g_utt2wav[utt] for utt in utt_list]  # Use global g_utt2wav
    text_list = [g_utt2text[utt] for utt in utt_list]  # Use global g_utt2text
    spk_list = [g_utt2spk[utt] for utt in utt_list]  # Use global g_utt2spk
    uttembedding_list = [g_utt2embedding[utt] for utt in utt_list]  # Use global g_utt2embedding
    spkembedding_list = [g_spk2embedding[g_utt2spk[utt]] for utt in utt_list]  # Use global g_spk2embedding and g_utt2spk
    speech_token_list = [g_utt2speech_token[utt] for utt in utt_list]  # Use global g_utt2speech_token

    # Save to parquet, utt2parquet_file, spk2parquet_file (Original comment in Chinese, kept for context if needed by user)
    # The comment translates to: "Save to parquet, utt2parquet_file, spk2parquet_file"
    df = pd.DataFrame()
    df['utt'] = utt_list
    df['wav'] = wav_list
    df['audio_data'] = data_list
    df['text'] = text_list
    df['spk'] = spk_list
    df['utt_embedding'] = uttembedding_list
    df['spk_embedding'] = spkembedding_list
    df['speech_token'] = speech_token_list
    df.to_parquet(parquet_file)
    with open(utt2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)
    logging.info('spend time {}'.format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()

    utt2wav, utt2text, utt2spk = {}, {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/text'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open('{}/utt2spk'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir))
    spk2embedding = torch.load('{}/spk2embedding.pt'.format(args.src_dir))
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir))
    utts = list(utt2wav.keys())

    # Prepare arguments for worker initializer
    initargs_tuple = (utt2wav, utt2text, utt2spk, utt2embedding, spk2embedding, utt2speech_token)

    # Using process pool to speedup
    pool = multiprocessing.Pool(processes=args.num_processes,
                                initializer=init_worker,
                                initargs=initargs_tuple)
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.tar'.format(i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file))
    pool.close()
    pool.join()

    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2, \
            open('{}/spk2data.list'.format(args.des_dir), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
