"""
Shazam-style audio fingerprinting — core engine (v2)

Database structure (database.pkl):
{
    'hashes': { (f1, f2, dt): [(song_name, t_anchor), ...], ... },
    'songs': {
        song_name: {
            'peaks': [(t_idx, f_idx), ...],
            'n_frames': int,
            'n_hashes': int,
            'duration_sec': float,
        },
        ...
    }
}
"""
import numpy as np
import librosa
from scipy import signal as sp_signal
from scipy.ndimage import maximum_filter
import os
from collections import defaultdict

SONG_DIR = "songs_db"
SR = 22050
NPERSEG = 1024
NOVERLAP = 512
N_PEAKS = 30
FAN_OUT = 10
DT_MIN = 1
DT_MAX = 40
DF_MAX = 200


def load_audio(path, sr=SR, duration=None):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return y


def make_spectrogram(y, sr=SR):
    f, t, Sxx = sp_signal.spectrogram(y, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sdb = (10 * np.log10(Sxx + 1e-10)).astype(np.float32)
    return f, t, Sdb


def find_peaks(Sdb, n_peaks=N_PEAKS):
    local_max = maximum_filter(Sdb, size=(20, 20))
    peaks_mask = (Sdb == local_max)
    peaks = []
    n_time = Sdb.shape[1]
    for t_idx in range(n_time):
        col_peaks = np.where(peaks_mask[:, t_idx])[0]
        if len(col_peaks) == 0:
            continue
        col = Sdb[:, t_idx]
        ranked = col_peaks[np.argsort(col[col_peaks])[::-1]][:n_peaks]
        for f_idx in ranked:
            peaks.append((t_idx, int(f_idx)))
    peaks.sort()
    return peaks


def hash_peaks(peaks):
    hashes = []
    for i, (t1, f1) in enumerate(peaks):
        count = 0
        for j in range(i + 1, len(peaks)):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt < DT_MIN:
                continue
            if dt > DT_MAX:
                break
            if abs(f2 - f1) > DF_MAX:
                continue
            hashes.append(((f1, f2, dt), t1))
            count += 1
            if count >= FAN_OUT:
                break
    return hashes


def build_database(song_dir=SONG_DIR, progress_callback=None):
    all_entries = []
    song_meta = {}
    songs = sorted([f for f in os.listdir(song_dir) if f.endswith('.mp3')])
    song_name_list = []

    for i, fname in enumerate(songs):
        name = os.path.splitext(fname)[0]
        path = os.path.join(song_dir, fname)
        y = load_audio(path)
        duration_sec = len(y) / SR
        _, t_axis, Sdb = make_spectrogram(y)
        peaks = find_peaks(Sdb)
        hashes = hash_peaks(peaks)

        song_idx = len(song_name_list)
        song_name_list.append(name)

        for (f1, f2, dt), t_anchor in hashes:
            all_entries.append((f1, f2, dt, song_idx, t_anchor))

        song_meta[name] = {
            'peaks': np.array(peaks, dtype=np.int32),
            'n_frames': Sdb.shape[1],
            'n_hashes': len(hashes),
            'duration_sec': duration_sec,
            'filename': fname,
        }
        del y, Sdb

        if progress_callback:
            progress_callback(i + 1, len(songs), name)
        else:
            print(f"  [{i+1}/{len(songs)}] {name}  ({len(peaks)} peaks, {len(hashes)} hashes)")

    dtype = np.dtype([
        ('f1', np.int16), ('f2', np.int16), ('dt', np.int16),
        ('song_idx', np.int16), ('t_anchor', np.int16)
    ])
    arr = np.array(all_entries, dtype=dtype)
    sort_keys = arr['f1'].astype(np.int64)*512*41 + arr['f2'].astype(np.int64)*41 + arr['dt']
    arr = arr[np.argsort(sort_keys)]

    return {
        'lookup': arr,
        'song_names': song_name_list,
        'songs': song_meta,
    }


def _hash_key(f1, f2, dt):
    return int(f1)*512*41 + int(f2)*41 + int(dt)


def match(query_audio, db, top_k=5):
    """
    Returns: ranked (list of (song_name, score)), histograms (dict), peaks, hashes,
             song_offsets (dict song -> list of offsets, for the alignment-spike plot)
    """
    _, _, Sdb = make_spectrogram(query_audio)
    peaks = find_peaks(Sdb)
    query_hashes = hash_peaks(peaks)

    arr = db['lookup']
    song_names = db['song_names']
    key_arr = arr['f1'].astype(np.int64)*512*41 + arr['f2'].astype(np.int64)*41 + arr['dt']

    song_offsets = defaultdict(lambda: defaultdict(int))
    for (f1, f2, dt), t_q in query_hashes:
        key = _hash_key(f1, f2, dt)
        lo = np.searchsorted(key_arr, key)
        hi = np.searchsorted(key_arr, key, side='right')
        if lo == hi:
            continue
        for row in arr[lo:hi]:
            name = song_names[row['song_idx']]
            song_offsets[name][int(row['t_anchor']) - t_q] += 1

    scores = {}
    histograms = {}
    for song_name, hist in song_offsets.items():
        if not hist:
            continue
        peak_count = max(hist.values())
        if peak_count < 3:
            continue
        scores[song_name] = peak_count
        histograms[song_name] = hist

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k], histograms, peaks, query_hashes, song_offsets


def best_offset(histograms, song_name):
    if song_name not in histograms:
        return None
    hist = histograms[song_name]
    if not hist:
        return None
        
    if isinstance(hist, tuple) and len(hist) == 2:
        counts, bins = hist
        peak_idx = np.argmax(counts)
        return (bins[peak_idx] + bins[peak_idx + 1]) / 2

    best_off = max(hist, key=hist.get)
    return best_off
