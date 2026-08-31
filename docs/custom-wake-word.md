# Training the "Hey Gary" wake word

openWakeWord only ships a few pretrained phrases, so a custom wake word means
training a tiny model. The project provides an automatic Colab notebook that
does everything (synthetic speech generation → training → export) on a free
GPU. Budget ~15 minutes of clicking and ~30–60 minutes of waiting.

**Train "hey gary", not bare "gary"** — one-word wake phrases false-trigger
badly (any "Gary" on TV wakes the house).

## Steps

1. Open the official training notebook in Colab (Google account needed):
   <https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb>
2. Runtime → Change runtime type → **GPU** (T4 is fine).
3. In the config cell, set the target phrase to `hey gary`. Defaults are fine
   everywhere else on a first attempt.
4. Runtime → Run all. Wait. The final cells produce `hey_gary.onnx`
   (sometimes named after the phrase) for download.
5. Drop the file into this repo as `models/hey_gary.onnx`.
6. In `.env`:

   ```
   WAKE_MODEL=models/hey_gary.onnx
   ```

7. Validate it like any wake model — the spike protocol:

   ```powershell
   uv run scripts/m3_spike.py --monitor 10    # quiet + with music, no phrase said
   ```

   then say "hey Gary" ~10× from across the room. Tune `WAKE_THRESHOLD`
   (custom models sometimes want 0.4–0.6) until false accepts are rare and
   misses are rarer.

## Notes

- The assistant's persona name is separate: `ASSISTANT_NAME` in `.env`
  (already Gary). Until the model is trained, the wake phrase stays
  "hey jarvis" — Gary answering to Jarvis's name is temporary and harmless.
- If the first model is mediocre (it happens with synthetic-only data), the
  notebook has knobs for more training clips; re-runs are cheap.
