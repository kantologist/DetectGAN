from __future__ import print_function

import numpy as np

# Constants
VAD_THRESHOLD = -80   # if a frame has no filter that exceeds this threshold, it is assumed silent and removed
VAD_MIN_NFRAMES = 150 # if a filtered utterance is shorter than this after VAD, the full utterance is retained


def bulk_VAD(feats, verbose=False):
    if verbose:
        original_avg_len = sum([len(utt) for utt in feats]) / len(feats)

    feats = [normalize(VAD(utt, verbose=verbose)) for utt in feats]

    if verbose:
        new_avg_len = sum([len(utt) for utt in feats]) / len(feats)
        #print("Average # frames before VAD:", original_avg_len, "; after VAD:", new_avg_len)

    return feats


def normalize(utterance):
    utterance = utterance - np.mean(utterance, axis=0, dtype=np.float64)
    utterance = np.float16(utterance)

    return utterance


def VAD(utterance, threshold=VAD_THRESHOLD, min_nframes=VAD_MIN_NFRAMES, verbose=False):
    if not threshold:
        return utterance

    filtered = utterance[utterance.max(axis=1) > threshold]

    if len(filtered) < min_nframes:
        if verbose:
            print("Skipping VAD for utterance with post-filtering length", len(filtered), "frames")
        return utterance

    return filtered

