## 🔬 Question 1: Frequency Forensics & Digital Detective

### Q1A: 'The Ghost Signal' (Frequency-Domain Image Recovery)
* **Objective:** Isolate and eliminate malicious periodic interference corrupting a grayscale reconnaissance image.
* **Methodology:**
  * Map the 2D spatial signal into the frequency domain using the Discrete Fourier Transform (DFT).
  * Shift the low frequencies to the center of the transform to analyze the magnitude spectrum on both linear and dB scales.
  * Identify high-amplitude localized spike fingerprints caused by periodic noise.
  * Design a suitable frequency-domain filter to suppress the unwanted noise frequencies while preserving the critical text details.
  * Reconstruct the filtered image using the Inverse DFT to successfully recover the hidden message.

### Q1B: 'Missing Boundaries' (2D Convolution & Edge Extraction)
* **Objective:** Extract structural object boundaries and track rapid intensity changes within a scene.
* **Methodology:**
  * Explore the relationship between derivatives, gradients, and edge detection using 2D convolution.
  * Apply a spatial Sobel filter kernel to compute horizontal and vertical intensity gradients.
  * Analyze the destructive impact of spatial noise on detected boundaries and evaluate the role of pre-smoothing operations to preserve weak edges.

---

## 🫀 Question 2: The Midnight Episode ('Catching the Arrhythmia')
This section focuses on designing an automated verification pipeline to scan continuous Holter monitor ECG recordings and precisely flag cardiac anomalies.

### Key Areas Explored:
* **Signal Property Auditing:** Evaluated a discrete-time signal to calculate its exact clip length, fundamental frequency, resting heart rate (BPM), and the sample footprint of a single cardiac cycle.
* **Frequency Band Analysis:** Profiled the ECG magnitude spectrum to differentiate the high-frequency contributions of sharp QRS complexes from smooth P/T waves, and modeled spectral changes during tachycardia.
* **Optimal Windowing Strategies:** Investigated windowing constraints using rectangular windows to isolate a single healthy cycle, documenting the classic time-frequency resolution trade-off.
* **Normalized Shape Correlation:** Built a scale-invariant template-matching engine using cross-correlation to address amplitude drift and baseline wander while easily flagging inverted waveform anomalies.
* **Arrhythmia Onset & Spectrogram Tracking:** Developed threshold detection rules to flag the exact onset of an episode, and evaluated how rolling Short-Time Fourier Transforms (Spectrograms) change visually between healthy and arrhythmia regions.
* **Sampling & Aliasing Constraints:** Analysed the clinical dangers of downsampling near signal information thresholds by applying the Nyquist theorem, exploring both anti-aliasing fixes and their associated engineering costs.

##  Q3A/B — Audio Fingerprinting Identifier (v2)

### Files
- `app.py` — Streamlit app: Library / Identify / Batch tabs.
- `fingerprint.py` — core engine: spectrogram, constellation peaks, paired-hash fingerprinting, database building, matching, and `best_offset()` (used to position the highlighted window in the full-song reconstruction view).
- `database.pkl` — pre-built database for all 50 songs (67 MB): per-hash lookup table **plus** each song's full constellation and metadata (duration, peak count, hash count) — the richer structure needed for the Library tab and the full-song fingerprint reconstruction view.
- `samples/sample1.mp3` … `sample5.mp3` — real 30-second clips cut from 5 different songs in the database, for the "Try a sample" feature.
- `requirements.txt` — Python (pip) dependencies.
- `packages.txt` — system-level (apt) dependencies — `ffmpeg` + `libsndfile1`, needed for MP3 decoding on Streamlit Cloud.

### Running locally
```bash
pip install -r requirements.txt
streamlit run app.py

### Re-building the database from scratch
```bash
import fingerprint as fp, pickle
db = fp.build_database()   # expects songs_db/ folder with the 50 .mp3 files
with open('database.pkl', 'wb') as f:
    pickle.dump(db, f)
