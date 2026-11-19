# Changelog

## 0.1.0

First release.

- `Identity` API: `enroll`, `who_is_here`, `on_person_arrived`, `on_person_left`, `start`, `stop`,
  `remember`, `recall`, `forget`, `export`.
- Face recognition (YuNet + SFace) and voice recognition (Silero VAD + ECAPA), fused into one
  confidence.
- Head-turn liveness during enrolment and recognition.
- Encrypted SQLite storage of face/voice embeddings and memory text; raw images and audio are never
  stored; hard delete on `forget`.
- The OS keyring call times out after 5 s and falls back to a 0600 key file. If a database already
  exists and the keyring is unresponsive with no key file, cheta refuses instead of creating a new
  key that would orphan the data.
- Per-person semantic memory with optional expiry. Memory text and memory embeddings are both
  encrypted; search decrypts one person's memories and ranks them in NumPy (no sqlite-vec).
- Every model is pinned to the exact Hugging Face commit cheta was tested with.
- `Identity(single_modality_factor=...)` sets the confidence penalty for a lone face or voice.
- `cheta calibrate DIR` recommends a `voice_threshold` from your own recordings.
- Reachy Mini support (`cheta[robot]`) and `examples/greeter.py`.
- CLI: `cheta enroll | list | forget | demo | listen | remember | recall`.
