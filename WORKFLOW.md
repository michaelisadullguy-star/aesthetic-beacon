# Aesthetic Beacon: workflow guide

How to run each step, what it expects and what it tells you. The file formats
are described in [data/README.md](data/README.md).

## Set-up

```bash
pip install -r requirements.txt
```

- **Python** 3.10 or newer. `torch`, `numpy`, `scipy` and `Pillow` are needed
  for everything; `pyserial` only to read the headset live from a serial port.
- **Processing** 4 for the display. No extra library is required.
- All commands are run from the repository root. Add `--config my.json` to use
  your own settings; a configuration file only needs the values it changes
  (see `config.example.json`).

## One command

```bash
python src/workflow_orchestrator.py --all
```

runs Steps 1–5. A step is skipped when neither its inputs nor its settings
changed since it last succeeded (`--force` redoes everything). Add
`--source …` to continue straight into the live loop. If something is missing,
the run stops with a message saying what and where — `--check` shows the same
overview without running anything.

`--quick` trains small models in a fraction of the time; use it to try things
out, not for the models you keep.

---

## Step 1 — convert the headset logs

```bash
python src/workflow_orchestrator.py --step 1 --input path/to/logs
```

Decodes every `.txt` / `.hex` / `.bin` recording in the folder (default
`data/step1_hex_eeg`) and writes its consecutive five-second segments as
`<name>_segment_0000.npy`, each an 8×3200 matrix of band amplitudes.

Read the summary line of each file:

```
session_001.txt: 59.8 s → 11 segments (10 clean)  [device runs at 512 Hz, resampled]
```

- *clean* segments pass the quality gate (contact, clipping, flat line, artefacts).
- *device runs at … Hz* appears when the measured rate differs from 640 Hz.
- *corrupt packets skipped* counts checksum failures; a few are normal over Bluetooth.
- *poor headset contact* means the headset itself reported bad contact for more
  than a tenth of the recording.

The other steps read hex logs directly, so Step 1 is not a prerequisite for
them. Its value is the quality report — run it on new recordings before
relying on them.

```python
from src.pipeline import iter_brainwave_segments

for segment in iter_brainwave_segments(open("session.txt").read()):
    print(segment.shape)        # (8, 3200)
```

## Step 2 — the to-viewer model

```bash
python src/workflow_orchestrator.py --step 2 --train
```

Learns to tell enjoyed moments of listening from the rest of the same session.
`annotations.json` lists each listening session with its EEG log, the audio
file and the moments the listener named afterwards. Five-second windows around
those moments are the positive examples; an equal number of windows elsewhere
in the session, clear of every named moment, are the controls.

The model sees EEG and audio, but for a random half of the training examples
the audio is withheld, so it also learns to judge from EEG alone — which is
how Step 3 uses it.

Reported: accuracy and AUC on held-out listening moments, and `auc_eeg_only`
for the EEG-only mode. An `auc_eeg_only` near 0.5 means the reading scores of
Step 3 will be noise.

## Step 3 — which passages did readers enjoy?

```bash
python src/workflow_orchestrator.py --step 3
```

Slides the viewer model over each reader's recording (five-second windows, one
second apart) and writes `<reader>_enjoyment.json`. If `article.txt` is
present, the windows are assigned to passages — paragraphs separated by blank
lines — and the passages are ranked in `passage_scores.json`.

- Which passage a reader was on is estimated from a steady reading speed,
  dividing the recording among the passages by their length. If you know
  better, give `passage_times` (or just `reading_start` / `reading_end`) per
  reader in `sessions.json`.
- Each reader's scores are standardised against their own average before
  readers are combined, so one generally enthusiastic reader does not decide
  the ranking.
- The top 30 % of passages are marked *excellent* (★) and equally many at the
  bottom become the *controls* (·); `step3.excellent_fraction` changes that.

`--input path/to/one_recording.txt` scores a single recording.

## Step 4 — the to-creator model

```bash
python src/workflow_orchestrator.py --step 4 --train
```

Labels come from two places and are combined:

1. **Automatic.** For every writing session described in
   `step4_creator_training/annotations/` (or recorded by Step 6 with
   `--watch-document`) whose article is the one ranked in Step 3: the keystroke
   log gives the time each passage was written; time spent on excellent
   passages is labelled *creatable*, time spent on control passages *not
   creatable*. The result is saved under `annotations/derived/`, so it stays
   part of the training data when you move on to the next article.
2. **Manual.** Stretches you marked yourself under `excellent_passages`.

Each labelled stretch is cut into overlapping five-second windows. Band levels
are expressed relative to the creator's **reference profile**: their typical
band levels across all training sessions, stored inside the model. For each
session the profile is shifted by that session's overall signal level
(electrode contact) and allowed to follow the session's own band levels by a
quarter (`baseline_adaptation` in `src/eeg_features.py`). Training windows and
live windows are computed the same way. Anchoring to a fixed profile is what
lets the model tell an hour of rest from an hour of flow; a baseline taken
from the session alone would make either look "normal".

Reported: accuracy and AUC on held-out stretches. Below an AUC of 0.6 a warning
tells you the live loop will fire rarely or at random.

## Step 5 — EEG to Processing parameters

```bash
python src/workflow_orchestrator.py --step 5 --train
```

Pairs EEG recorded while looking at an artwork with the parameter set that
makes the sketch echo that artwork. From each recording the middle five seconds
are used (plus four windows shifted by up to a second), avoiding the eye-opening
at the start and the anticipation at the end.

The parameters of an artwork come from, in order of preference:

1. `metadata.json` next to the artworks — your own values;
2. a CNN trained with `python -m src.train_processing_features` (`step5.processing_weights`);
3. direct analysis of the image: the largest colour area becomes the
   background, the next four the palette, and contrast, edge density, grain
   and edge directions set the texture parameters.

To see and edit what the analysis produces:

```bash
python tools/export_cnn_predictions_as_metadata.py --artworks-dir data/step5_image_training/artworks --output data/step5_image_training/metadata.json
```

Reported: `skill`, comparing the held-out error with always predicting the
average parameter set. 0 means the EEG adds nothing; with few artworks expect
it to be low. The model starts from the average set, so a weak model shows a
steady, average texture rather than noise.

## Step 6 — the live loop

```bash
python src/workflow_orchestrator.py --step 6 --realtime --source serial:/dev/ttyUSB0
```

Start the sketch (`scripts/processing_templates/dynamic_texture_osc`, **Run**)
either before or after; `--launch-sketch` starts it for you if
`processing-java` is on the PATH.

What you see in the terminal:

```
Headset sampling rate: 512 Hz (measured from the headset's status packets); resampling to 640 Hz
[    5.0s] state: CALIBRATING (5s of 20s)
[   45.0s] state: CREATABLE
[   45.0s] p(creatable)=0.93  smoothed=0.72  → parameters updated
[   90.0s] state: IDLE
[   90.0s] p(creatable)=0.08  smoothed=0.31  · holding
[  120.0s] state: BAD SIGNAL (poor_contact)
```

| State | Meaning | Display |
|-------|---------|---------|
| calibrating | measuring the session's signal level (first 20 s of clean signal) | default texture |
| creatable | probability above the threshold | new parameters every interval |
| idle | not creatable | keeps the last parameters |
| bad signal | the window failed the quality gate | keeps the last parameters |
| no signal | nothing received for 3 s; reconnecting | keeps the last parameters |

Useful options: `--duration 600` stops after ten minutes, `--no-osc` writes the
parameter file only, `--no-record` keeps nothing, `--watch-document draft.txt`
logs your writing, `--record-only` records without models or display.

Settings (`step6` in the configuration):

| Setting | Default | Effect |
|---------|---------|--------|
| `creatable_threshold` | 0.6 | probability at which the state is entered |
| `release_threshold` | threshold − 0.1 | probability below which it is left |
| `probability_smoothing` | 0.6 | weight of the newest window; 1 = no smoothing |
| `parameter_smoothing` | 0.6 | how far each update moves towards the new prediction |
| `update_interval` | 5.0 | seconds between evaluations (windows are always 5 s) |
| `calibration_seconds` | 20 | clean signal needed before the first decision |

Lower `update_interval` (for example to 1) for a more responsive beacon; the
windows then overlap.

### From your own code

```python
from src.model import load_checkpoint
from src.pipeline import run_feedback_loop

creator, _ = load_checkpoint("data/step6_realtime/models/creator_model.pt", kind="creator")
parameters, _ = load_checkpoint("data/step6_realtime/models/parameter_model.pt", kind="parameter")

def show(parameter_set, is_creatable):
    print(is_creatable, parameter_set["cellSize"])

run_feedback_loop(
    my_band_matrices,            # any iterable of (8, 3200) arrays
    creator_model=creator,
    parameter_model=parameters,
    update_callback=show,
    creatable_threshold=0.6,
)
```

`run_feedback_loop` takes prepared band matrices. For the complete chain from
raw bytes use `src.realtime.RealtimeSession`, as the orchestrator does.

---

## Connecting the headset

- **USB dongle / Bluetooth serial port:** `--source serial:/dev/ttyUSB0`
  (Linux), `serial:/dev/tty.<name>` (macOS), `serial:COM3` (Windows). The
  default is 57600 baud; append `@115200` or similar if your module differs.
- **You already log with another tool:** let it write its hex log and use
  `--source follow:/path/to/that/log.txt`.
- **Fixed sampling rate:** if the stream carries no status packets and the rate
  cannot be measured, set `features.device_sample_rate` in the configuration.
- **Mains hum:** 50 Hz is filtered out by default because it lies inside the γ2
  band. Set `features.notch_hz` to 60 where the mains run at 60 Hz, or to
  `null` to disable. Changing it requires retraining.

## Troubleshooting

**`Cannot continue: …`** — every step stops with a message naming the missing
file or the malformed entry. `--check` gives the overview.

**Step 1 says "no headset packets found — decoded as legacy 8-channel dump".**
The file does not contain `AA AA …` packets. It is treated as eight interleaved
16-bit channels, the format older versions of this repository assumed.

**All windows are "bad signal".** Look at the reason in brackets:
`poor_contact` — the headset reports bad contact, re-seat the electrode and ear
clip; `flatline` — no signal at all; `clipping` — the amplifier saturates;
`artifact` — strong movement.

**The state never becomes creatable.** Check the creator model's AUC in
`creator_model.history.json`. If it is good, lower `creatable_threshold`
slightly; if it is poor, record more writing sessions.

**The sketch does not react.** Press **O**: the overlay shows whether the UDP
port is open and when the last update arrived, and through which route. "UDP
unavailable" usually means a second copy of the sketch is running. If the
sketch was moved out of the repository, set `LIVE_PARAM_FILE` at its top to the
absolute path of `current_parameters.json`.

**The sketch window stays black or does not open.** Change `RENDERER` at the
top of the sketch from `P2D` to `JAVA2D`.

**Changing a parameter's range or adding one.** Edit `src/parameters.py`, run
`python -m src.parameters` to regenerate the sketch's schema file, and retrain
Step 5.
