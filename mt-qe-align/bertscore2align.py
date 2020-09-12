#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = \
'''
    Generating alignments for every pair of sentences in the given MT&PE corpus files, using *BERT's representation.
    The BERT model is expected to be prepared in advance in model-dir, specifically: config.json, vocab.txt and 
    pytorch_model.bin are expected.
    
    In particular, cosine similarity is calculated and only pair of words whose cosine similarity is above a
    specified threshold will be extracted as an aligned match.
'''

import argparse
import itertools
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask


def sent_encode(tokenizer, sent):
    "Encoding as sentence based on the tokenizer"
    sent = sent.strip()

    def map_offset(orig_tokens, pieced_tokens):
        mapping = {}
        pieced_i = 0
        for orig_i, orig_token in enumerate(orig_tokens):
            tmp_token = pieced_tokens[pieced_i]

            # normalize orig_token (lowercase, accent-norm. No punc-split-norm)
            orig_token = tokenizer.basic_tokenizer._run_strip_accents(orig_token).lower()

            while True:
                mapping[pieced_i] = orig_i
                pieced_i += 1
                if tmp_token == orig_token:
                    break
                else:
                    tmp_token += pieced_tokens[pieced_i].replace('##', '')

                if len(tmp_token) > len(orig_token):
                    msg = f'Original Text:  {" ".join(orig_tokens)}\n' \
                          f'Pieced Text: {" ".join(pieced_tokens)}\n' \
                          f'Original Token: {orig_token}\n' \
                          f'Pieced Tmp Token: {tmp_token}\n' \
                          f'Mapping: {mapping}'
                    raise ValueError('Maybe original tokens and pieced tokens does not match.\n' + msg)

        return mapping

    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    else:
        pieced_token_ids = tokenizer.encode(sent, add_special_tokens=True)

        # we need a tokenize method which do not add UNK but still do wordpiece tokenization
        pieced_tokens = []
        for token in tokenizer.basic_tokenizer.tokenize(sent, never_split=tokenizer.all_special_tokens):
            wp_token = tokenizer.wordpiece_tokenizer.tokenize(token)
            if '[UNK]' in wp_token:
                assert len(wp_token) == 1, f'Token {token} is splited by wordpiece but still contains UNK??'
                pieced_tokens.append(token)
            else:
                pieced_tokens.extend(wp_token)
        assert len(pieced_tokens) + 2 == len(pieced_token_ids), 'Token list used for offset mapping does not match ' \
                                                                'which is input into *BERT.'

        piece_offset_mapping = map_offset(sent.split(), pieced_tokens)
        return pieced_token_ids, piece_offset_mapping


def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        out = model(x, attention_mask=attention_mask)
    emb = out[0]
    return emb


def pad_batch_stats(sen_batch, stats_dict, device):
    stats = [stats_dict[s] for s in sen_batch]
    emb = stats
    emb = [e.to(device) for e in emb]
    lens = [e.size(0) for e in emb]

    pad_sequence = torch.nn.utils.rnn.pad_sequence
    emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)

    def length_to_mask(lens):
        lens = torch.tensor(lens, dtype=torch.long)
        max_len = max(lens)
        base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
        return base < lens.unsqueeze(1)

    pad_mask = length_to_mask(lens).to(device)
    return emb_pad, pad_mask


def greedy_cos_sim(ref_embedding, ref_masks, hyp_embedding, hyp_masks):
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())

    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    return sim, masks  # (batch_size, longest(in_batch)_hyp_len, longest(in_batch)_ref_len)


def process_sim(sim, masks, threshold, process_method=None):
    batch_size = sim.size(0)
    align_lines = []
    for b in range(batch_size):    # process sentence pairs one-by-one
        align = []
        sim_matrix = sim[b, 1:-1, 1:-1]    # exclude first row and col (CLS) and last row and col(SEP or PAD)
        longest_hyp_len, longest_ref_len = sim_matrix.shape

        mask = masks[b, 1:-1, 1:-1].type(torch.bool)

        if process_method is None:
            pass

        elif process_method == 'hungarian':
            cost_matrix = torch.ones_like(sim_matrix)
            cost_matrix -= sim_matrix
            aligned_hyp, aligned_ref = linear_sum_assignment(cost_matrix.cpu())
            for h, r in zip(aligned_hyp, aligned_ref):
                sim_matrix[h, r] = -float('inf')

        elif process_method == 'grow-diag-final':

            hyp2ref = torch.zeros_like(sim_matrix, device='cpu')
            for h, r in enumerate(sim_matrix.argmax(dim=1)):
                hyp2ref[h][r] = 1

            ref2hyp = torch.zeros_like(sim_matrix, device='cpu')
            for r, h in enumerate(sim_matrix.argmax(dim=0)):
                ref2hyp[h][r] = 1

            align_matrix = np.logical_and(hyp2ref, ref2hyp)
            union_matrix = np.logical_or(hyp2ref, ref2hyp)

            neighbours = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]

            def _grow_diag():
                point_added = False
                for hyp, ref in itertools.product(range(longest_hyp_len), range(longest_ref_len)):
                    if not align_matrix[hyp][ref]:
                        continue
                    for nh, nr in neighbours:
                        if hyp + nh < 0 or hyp + nh >= longest_hyp_len or ref + nr < 0 or ref + nr >= longest_ref_len:
                            continue
                        if (not align_matrix[hyp + nh, :].sum() > 0 or not align_matrix[:, ref + nr].sum() > 0) \
                            and union_matrix[hyp+nh, ref+nr]:
                            align_matrix[hyp + nh, ref + nr] = 1
                            point_added = True

                if point_added:
                    _grow_diag()

            def _final(matrix):
                for hyp, ref in itertools.product(range(longest_hyp_len), range(longest_ref_len)):
                    if (not align_matrix[hyp, :].sum() > 0 or not align_matrix[:, ref].sum() > 1) \
                        and matrix[hyp, ref]:
                        align_matrix[hyp, ref] = 1

            _grow_diag()
            _final(hyp2ref)
            _final(ref2hyp)

            for hyp, ref in itertools.product(range(longest_hyp_len), range(longest_ref_len)):
                if not align_matrix[hyp, ref]:
                    sim_matrix[hyp, ref] = -float('inf')

        else:
            raise ValueError(f'Invalid process method {process_method}')

        for i in range(longest_hyp_len):
            if i + 1 >= longest_hyp_len or mask[i + 1, :].sum() == 0:    # PAD
                break
            for j in range(longest_ref_len):
                if j + 1 >= longest_ref_len or mask[i, j + 1] == 0:    # PAD
                    break
                if sim_matrix[i, j] >= threshold:
                    align.append((i, j))

        align_lines.append(align)

    return align_lines


def adapt_offset(align_lines, hyp_offset_mappings, ref_offset_mappings):
    new_align_lines = []
    for i, align_line in enumerate(align_lines):
        hyp_offset_mapping = hyp_offset_mappings[i]
        ref_offset_mapping = ref_offset_mappings[i]

        try:
            new_align_lines.append(
                list(set(
                    (hyp_offset_mapping[x], ref_offset_mapping[y]) for x, y in align_line
                ))
            )
        except Exception as e:
            print(i, align_line)
            print(hyp_offset_mapping)
            print(ref_offset_mapping)
            raise e

    return new_align_lines


def bert_score_to_align(hyps, refs, args):
    assert len(hyps) == len(refs), 'Different number of candidates and references.'

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir)
    model.to('cuda')

    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(' ')), reverse=True)

    sentences = dedup_and_sort(refs + hyps)
    iter_range = range(0, len(sentences), args.batch_size)
    stats_dict = {}
    offset_mapping_dict = {}
    for batch_start in tqdm(iter_range, mininterval=0.5, ncols=100, desc='Encoding Sentences'):
        sen_batch = sentences[batch_start: batch_start + args.batch_size]
        sen_batch_res = [sent_encode(tokenizer, s) for s in sen_batch]
        sen_batch_tokens = [r[0] for r in sen_batch_res]
        sen_batch_offset_mappings = [r[1] for r in sen_batch_res]

        padded_sens, lens, mask = padding(sen_batch_tokens, tokenizer.pad_token_id, dtype=torch.long)
        padded_sens = padded_sens.to('cuda')
        mask = mask.to('cuda')

        with torch.no_grad():
            batch_embedding = bert_encode(
                model, padded_sens, attention_mask=mask)

        for i, sen in enumerate(sen_batch):
            sequence_len = mask[i].sum().item()
            emb = batch_embedding[i, :sequence_len]
            stats_dict[sen] = emb
            offset_mapping_dict[sen] = sen_batch_offset_mappings[i]

    iter_range = range(0, len(refs), args.batch_size)
    with torch.no_grad():
        align_lines = []
        for batch_start in tqdm(iter_range, mininterval=0.5, ncols=100, desc='Calculating Similarity'):
            batch_refs = refs[batch_start: batch_start + args.batch_size]
            batch_hyps = hyps[batch_start: batch_start + args.batch_size]
            ref_stats = pad_batch_stats(batch_refs, stats_dict, device='cuda')
            hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device='cuda')
            ref_offset_mappings = [offset_mapping_dict[s] for s in batch_refs]
            hyp_offset_mappings = [offset_mapping_dict[s] for s in batch_hyps]

            sim, masks= greedy_cos_sim(*ref_stats, *hyp_stats)

            batch_align_lines = process_sim(sim, masks, args.sim_threshold, args.sim_process_method)

            batch_align_lines = adapt_offset(batch_align_lines, hyp_offset_mappings, ref_offset_mappings)
            align_lines.extend(batch_align_lines)

    return align_lines


def parse_args():
    parser = argparse.ArgumentParser(NOTE)

    parser.add_argument('-mt', '--machine-translation',
                        help='Path to the MT corpus.')
    parser.add_argument('-pe', '--post-edit',
                        help='Path to the PE corpus.')
    parser.add_argument('-d', '--model-dir',
                        help='Directory where pytorch_model.bin, config.json and vocab.txt depicting the specific '
                             '*BERT model are saved.')
    parser.add_argument('-o', '--output',
                        help='Path to the output file.')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size when encoding sentences. DEFAULT: 64.')
    parser.add_argument('--sim-threshold', type=float, default=0.5,
                        help='Similarity score above which is regarded to be a possible alignment. DEFAULT: 0.5')
    parser.add_argument('--sim-process-method', default=None, choices=['hungarian', 'grow-diag-final'],
                        help='Some process methods to filter out invalid alignments before filter them by threshold.')
    parser.add_argument('--mt-to-pe', action='store_true',
                        help='In default settings, output of this script is a PE-to-MT alignment file. If the '
                             'opposite alignment is wanted, then add this option.')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    with open(args.machine_translation, 'r', encoding='utf-8') as f:
        mt_lines = [l.strip() for l in f]

    with open(args.post_edit, 'r', encoding='utf-8') as f:
        pe_lines = [l.strip() for l in f]

    align_lines = bert_score_to_align(mt_lines, pe_lines, args)

    with open(args.output, 'w', encoding='utf-8') as f:
        for align_line in align_lines:
            if args.mt_to_pe:
                key_func = lambda x:(x[0], x[1])
            else:
                key_func = lambda x:(x[1], x[0])

            align_line_str = ' '.join([f'{i}-{j}' if args.mt_to_pe else f'{j}-{i}' for i,j \
                                      in sorted(align_line, key=key_func)]) + '\n'
            f.write(align_line_str)


if __name__ == '__main__':
    main()
    # refs = ['28-Year-Old Chef Found Dead at San Francisco',
    #         'He was a kind spirit with a big heart.']
    #
    # hyps = ['a chef is found killed at San Francisco who is 26 years old',
    #         'He was a friendly person with a big heart.']
    #
    # align_lines = bert_score_to_align(hyps, refs, 'model', 2, 0.6)
    # for align_line in align_lines:
    #     align_line_str = ' '.join(f'{j}-{i}' for i,j \
    #                               in sorted(align_line, key=lambda x:x[1]))
    #     print(align_line_str)