# Changelog

## 0.1.0

First release.

- `Identity` API: `enroll`, `who_is_here`, `on_person_arrived`, `on_person_left`, `start`, `stop`,
  `remember`, `recall`, `forget`, `export`.
- Face recognition (YuNet + SFace) and voice recognition (Silero VAD + ECAPA), fused into one
  confidence.
- Head-turn liveness during enrolment and recognition.
- Encrypted SQLite storage; raw images and audio are never stored; hard delete on `forget`.
- Per-person semantic memory with optional expiry.
- Reachy Mini support (`cheta[robot]`) and `examples/greeter.py`.
- CLI: `cheta enroll | list | forget | demo | listen | remember | recall`.
