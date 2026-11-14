# Aesthetic Beacon

**An EEG-driven creative feedback loop.**

Aesthetic Beacon learns which of a creator's brain states go together with
writing that readers enjoy, and — while the creator works — turns the EEG into a
living texture on a small screen whenever such a "creatable" state is detected.

- **[Workflow guide](WORKFLOW.md)** — each step in detail, real-hardware set-up, troubleshooting
- **[Data layout](data/README.md)** — where recordings go and what the files look like
- **[Example configuration](config.example.json)** — every setting with its default

## Try it in five minutes (no headset needed)

```bash
pip install -r requirements.txt
```

```bash
python src/workflow_orchestrator.py --demo --quick
```

This writes a small simulated study to `demo_data/`, runs all six steps on it
and ends with a verdict: whether the passages that were simulated as enjoyable
were found from the readers' EEG, and how often the live loop recognised the
simulated creatable state. Without `--quick` the full-size models are trained
(about three minutes on a laptop CPU).

To watch the display react, open
`scripts/processing_templates/dynamic_texture_osc` in
[Processing](https://processing.org/download), press **Run**, and replay the
simulated session in real time:

```bash
python src/workflow_orchestrator.py --demo --quick --speed 1
```

## With real recordings

```bash
python src/workflow_orchestrator.py --check
```

```bash
python src/workflow_orchestrator.py --all --source serial:/dev/ttyUSB0
```

`--check` lists which inputs are present. `--all` trains everything that is
out of date and then starts the live loop on the given EEG source (`COM3`-style
names on Windows). Run it again whenever recordings are added: only the
affected steps are redone.

## The six steps

| Step | What happens | Input | Output |
|------|--------------|-------|--------|
| 1 | Decode the headset's hex logs into 8×3200 band matrices; report signal quality | hex logs | `.npy` segments, `conversion_report.json` |
| 2 | Train the **to-viewer** model: was this moment of listening enjoyed? | listeners' EEG + music + the moments they named | `viewer_model.pt` |
| 3 | Score readers' EEG with that model and rank the article's passages | readers' EEG + the article | `passage_scores.json` |
| 4 | Train the **to-creator** model: label the creator's EEG by whether the passage being written turned out excellent | creator's EEG + keystroke log + Step 3 ranking | `creator_model.pt` |
| 5 | Train the EEG → **Processing parameters** model | EEG recorded while viewing artworks + the artworks | `parameter_model.pt` |
| 6 | **Live loop**: every 5 s, if the creator is in a creatable state, refresh the 50 display parameters from the EEG; otherwise keep the last set | live headset stream | OSC to the sketch, recorded session |

Step 4 needs no manual marking: the keystroke log says when each passage was
written, Step 3 says which passages readers enjoyed, and the two are joined
automatically. Hand-made annotations are still accepted.

## What the signal looks like

The headset has one forehead electrode and speaks the NeuroSky ThinkGear
protocol. Its logs are hexadecimal text:

```
AA AA 04 80 02 00 52 2B AA AA 04 80 02 00 47 36 ...
```

Each `AA AA 04 80 02 hi lo checksum` packet is one raw sample. The pipeline
verifies checksums, skips damaged bytes and splits the single channel into the
amplitude envelopes of eight bands — δ 0.5–4, θ 4–8, α1 8–10, α2 10–13,
β1 13–20, β2 20–30, γ1 30–45, γ2 45–60 Hz — sampled at 640 Hz. Five seconds of
that is the 8×3200 matrix every model consumes.

**Sampling rate.** The models work at 640 Hz, but the device's rate is measured
from the stream (the headset sends one status packet per second) instead of
being assumed, and the signal is resampled if it differs. This matters: if a
512 Hz stream were read as 640 Hz, every band would sit 25 % too high. Step 1
reports the rate it found for each recording.

## The live loop

```bash
python src/workflow_orchestrator.py --step 6 --source serial:/dev/ttyUSB0
```

| Source | Meaning |
|--------|---------|
| `serial:<port>[@baud]` | the headset's serial port (default 57600 baud) |
| `follow:<file>` | a hex log that another program is writing right now |
| `replay:<file>` | a finished recording (`--speed 0` = as fast as possible) |
| `tcp:<host>:<port>` | a socket delivering the raw bytes |
| `sim` | the built-in simulator |

What makes it safe to leave running:

- **Bad signal is never classified.** Windows with poor electrode contact,
  clipping, a flat line or heavy movement artefacts are skipped and the display
  keeps its parameters.
- **A dropped connection is survived.** The source reconnects by itself; no
  window ever mixes signal from before and after the gap.
- **No flicker.** The state is entered at probability 0.6 and left below 0.5,
  on a smoothed probability; parameters glide from one set to the next.
- **Nothing unsafe reaches the sketch.** All 50 parameters are clamped to
  ranges defined in `src/parameters.py`, and the sketch clamps again.
- **Calibration.** The first 20 s establish the session's signal level.
  Band levels are then judged against the creator's own reference profile
  (learned in Step 4), corrected for that level, so neither a change in
  electrode contact nor a long stretch in one state throws the detector off.
- **Every session is recorded** under `data/step6_realtime/sessions/` — the EEG
  in the headset's own format plus a log of every decision — ready to be used
  as training data.

## The display

`scripts/processing_templates/dynamic_texture_osc` is the sketch for the
1080×640 screen. It needs no extra Processing library. It listens for OSC on
UDP port 12000 and also watches
`data/step6_realtime/outputs/current_parameters.json`, so it keeps working if
UDP is blocked. Keys: **O** status overlay, **R** reload defaults, **S** save
frames as PNG.

`dynamic_texture_template` is the static version of the same texture, driven
only by its JSON file.

## Closing the loop

1. **Write** with the headset on. The first time, before any model exists:
   ```bash
   python src/workflow_orchestrator.py --step 6 --record-only --source serial:/dev/ttyUSB0 --watch-document draft.txt
   ```
   Later, drop `--record-only` to write with the display. `--watch-document`
   logs how the text grows each time your editor saves it.
2. **Let people read it.** Copy the session's `article.txt` to
   `data/step3_article_reading/article.txt` and put the readers' hex logs in
   `data/step3_article_reading/eeg/`.
3. **Retrain:** `python src/workflow_orchestrator.py --all`. The recorded
   session is recognised by its article, labelled from the new ranking and
   added to the creator model's training data.

## Project structure

```
src/
  thinkgear.py             headset packet decoding
  eeg_features.py          band envelopes, quality gate, baselines, recordings
  parameters.py            the 50 display parameters and their encoding
  pipeline.py              the six steps as functions
  model.py                 the three networks and their checkpoints
  training.py              training loop, validation split, calibration
  brainwave_dataset.py     Step 5 dataset
  processing_features.py   artwork → parameters (analysis, optional CNN)
  eeg_sources.py           live sources (serial, follow, replay, tcp)
  realtime.py              the live session
  osc_bridge.py            Python → Processing
  simulator.py             simulated headset and study
  workflow_orchestrator.py command line
scripts/processing_templates/   the Processing sketches
tests/                          pytest suite
```

## Tests

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest tests
```

The suite decodes the bytes shown in the project description, checks that a
live window is computed identically to a training window, runs the whole loop
on a simulated study, replays awkward sessions (all rest, minutes of
uninterrupted flow, a change of electrode contact, a dropped connection) and,
if a JDK is installed, compiles the sketch's OSC receiver and feeds it from the
Python bridge.

## Known limits

- **Not yet run against the physical headset.** Decoding is checked against the
  hex dump in the project description and the serial path against a
  pseudo-terminal; expect to adjust the port name and possibly the baud rate.
- **The simulator proves the plumbing, not the science.** It shows that
  relationships present in the data are recovered. Whether such relationships
  exist in real EEG is the open question of the project; each step reports
  held-out scores (AUC, skill) so you can tell.
- **Small data.** A handful of sessions will give unstable models. Step 4 warns
  when the creator model barely separates the two states.
- **One article at a time.** Before ranking a new article, move the previous
  article's reader logs out of `step3_article_reading/eeg/`.
- **The EDA sensor is not read.** Artefacts are handled from the EEG itself and
  the headset's contact-quality value.
