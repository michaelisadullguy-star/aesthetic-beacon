# Data layout

Everything the workflow reads and writes lives under this folder (or under
`data_root` in your configuration). The contents are not tracked by git.

To see the layout filled in, generate a simulated study:

```bash
python scripts/generate_test_data.py --output demo_data
```

## EEG recordings

Wherever a recording is expected, any of these is accepted:

| File | Content |
|------|---------|
| `.txt`, `.hex` | the headset's log as hexadecimal text: `AA AA 04 80 02 00 52 2B …` (spaces, line breaks, `0x` prefixes or no separators at all) |
| `.bin` | the same byte stream in binary |
| `.npy`, one dimension | raw signal already at 640 Hz |
| `.npy`, shape `(8, samples)` | band envelopes you computed yourself |

Hex logs are the normal case; nothing needs converting by hand.

## `step1_hex_eeg/`

Logs to convert and check with Step 1. Output goes to `converted/`:
`<name>_segment_0000.npy` … (each 8×3200) and `conversion_report.json`.

## `step2_viewer_training/`

```
eeg/            one log per listening session
audio/          the music that was played (.wav)
annotations/annotations.json
```

```json
{
  "sessions": [
    {
      "eeg_file": "step2_viewer_training/eeg/G1_1-4 25-09-11-2036.txt",
      "audio_file": "step2_viewer_training/audio/bach_bwv848.wav",
      "enjoyed": [[12.0, 18.5], 41.0, 55.5],
      "audio_offset": 0.0
    }
  ]
}
```

- `enjoyed` — the moments the listener named: a time in seconds from the start
  of the EEG recording, or a `[start, end]` interval.
- `audio_offset` — seconds to add to an EEG time to get the matching audio
  time, if the music did not start with the recording.
- Paths are relative to the data folder (or absolute).

Pre-cut five-second pairs are accepted as well:

```json
{"samples": [{"eeg_file": "…/clip_001.npy", "audio_file": "…/clip_001.wav", "enjoyed": true}]}
```

## `step3_article_reading/`

```
article.txt     the article; passages are separated by blank lines
eeg/            one log per reader
sessions.json   optional
results/        written by Step 3
```

Without `sessions.json` every log in `eeg/` is one reader who read the article
from the start to the end of the recording. Use it when that is not so:

```json
{
  "sessions": [
    {"eeg_file": "step3_article_reading/eeg/reader01.txt", "reading_start": 4.0, "reading_end": 212.0},
    {"eeg_file": "step3_article_reading/eeg/reader02.txt", "passage_times": [[0, 31.5], [31.5, 60.2], [60.2, 97.0]]}
  ]
}
```

Results: `<reader>_enjoyment.json` (a score per window) and
`passage_scores.json` (the ranking, with the `excellent` and `control`
passages by index, starting at 0).

Rank one article at a time: move the previous article's reader logs elsewhere
before adding a new `article.txt`.

## `step4_creator_training/`

```
eeg/            the creator's logs and keystroke logs
annotations/    one .json per writing session
annotations/derived/   labels produced automatically (do not edit; regenerated)
```

A session to be labelled automatically:

```json
{
  "session_id": "creator_session_001",
  "eeg_file": "step4_creator_training/eeg/creator_session_001.txt",
  "keystroke_log": "step4_creator_training/eeg/creator_session_001_keystrokes.jsonl",
  "article": "step3_article_reading/article.txt"
}
```

The keystroke log is JSON lines (or one JSON array). Each entry has a time and
how much had been written by then:

```json
{"t": 12.4, "length": 153}
{"t": 14.1, "text": "The whole document so far…"}
{"timestamp": 1757613000.2, "key": "a"}
```

- time: `t` in seconds from the start of the EEG recording, or `timestamp` as
  Unix time together with `"started_at"` (the recording's start) in the session file;
- progress: `length` (characters so far), `text` (the document so far) or
  `key` (one key press; `Backspace` counts backwards).

Instead of a keystroke log you can state directly when each passage was written:

```json
"writing_timeline": [{"passage": 0, "start_time": 0.0, "end_time": 95.2}, {"passage": 1, "start_time": 95.2, "end_time": 171.0}]
```

A session labelled by hand:

```json
{
  "eeg_file": "step4_creator_training/eeg/creator_session_002.txt",
  "excellent_passages": [
    {"start_time": 120.5, "end_time": 150.0, "label": "creatable"},
    {"start_time": 300.0, "end_time": 330.0, "label": "not_creatable"}
  ]
}
```

Both kinds may be mixed; the model needs examples of both labels.

## `step5_image_training/`

```
artworks/       artwork_01.png, artwork_02.jpg, …
brainwaves/     recordings made while viewing them
metadata.json   optional: your own parameters per artwork
```

A recording belongs to the artwork whose name it carries. Any of:

```
brainwaves/artwork_01.txt
brainwaves/artwork_01__anna.txt
brainwaves/anna/artwork_01.txt        (recommended: one folder per participant)
```

Recordings in the same folder are treated as one sitting and share a baseline,
which is what preserves the differences between artworks — so keep each
participant in their own folder. Recordings must last at least five seconds;
from longer ones (ten seconds is typical) the middle is used.

`metadata.json` maps artwork names to parameters; artworks without an entry
are analysed automatically:

```json
{
  "artwork_01": {
    "cellSize": 32,
    "noiseScale": 0.01,
    "backgroundHue": 210,
    "palette": [{"h": 210, "s": 65, "b": 95}, {"h": 35, "s": 85, "b": 96}]
  }
}
```

Missing parameters take their default; values outside the safe range are
clamped. The 50 parameter names and ranges are listed in
`scripts/processing_templates/dynamic_texture_osc/data/parameter_schema.json`.

## `step6_realtime/`

```
models/     viewer_model.pt, creator_model.pt, parameter_model.pt
            <model>.history.json   losses and held-out scores of the last training
            pipeline_state.json    which steps are up to date
outputs/    current_parameters.json   what the display is showing (the sketch watches this)
            workflow.log
sessions/<date-time>/
            eeg.txt           the session's EEG, in the headset's hex format
            events.jsonl      one line per evaluated window
            keystrokes.jsonl  how the watched document grew (with --watch-document)
            article.txt       the document at the end of the session
            session.json      settings, measured sampling rate, summary
```

A line of `events.jsonl`:

```json
{"t": 45.0, "state": "creatable", "is_creatable": true, "probability": 0.93, "smoothed_probability": 0.72,
 "updated": true, "reason": "", "quality": {"ok": true, "reason": "ok", "poor_fraction": 0.0, "clip_fraction": 0.0, "artifact_fraction": 0.01},
 "parameters": {"palette_1_h": 212.4, "cellSize": 28.0, "…": 0}}
```
