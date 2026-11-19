# cheta

On-device face and voice recognition, memory and forgetting for Reachy Mini apps.
Apache 2.0.

cheta lets an app know who is in front of the robot, remember facts about them, and forget them
completely when asked. Everything runs locally.

- **No raw images or audio are ever stored.** Only embeddings (numbers) are kept, encrypted.
- **Nothing biometric leaves the device.** Models are downloaded once; after that it works offline.
- **Consent is enforced.** `enroll()` refuses unless you pass `consent=True`.
- **Forgetting is a hard delete.** `forget(person)` removes the embeddings and memories and vacuums the database.

## Install

```bash
pip install cheta
pip install "cheta[robot]"   # adds the Reachy Mini SDK (Python 3.11+)
python -m cheta.models       # one-time download of the models from the Hugging Face Hub
```

Python 3.10 or newer. On Linux, text-to-speech in the greeter example needs `espeak-ng`.

## Quickstart

```python
from cheta import Identity

idt = Identity(app="my_app", camera="webcam")  # camera="reachy" with the robot

# Enrol someone, with their consent. They are asked to turn their head (liveness).
alice = idt.enroll("Alice", consent=True)

idt.remember(alice, "Alice is learning the cello", expires_in_days=90)

idt.on_person_arrived(lambda p: print("hello", p.name, p.confidence))
idt.on_person_left(lambda p: print("bye", p.name))
idt.start()  # background recognition loop

print(idt.who_is_here())  # Person or None
print(idt.recall(alice, "what instrument?", k=3))
print(idt.export(alice))  # everything stored about Alice

idt.forget(alice)  # hard delete
idt.stop()
```

### Command line

```bash
cheta enroll --name Alice
cheta list
cheta remember --name Alice "Alice is learning the cello"
cheta recall --name Alice "instrument"
cheta demo                  # live recognition in the terminal
cheta forget --name Alice
```

Global options: `--app`, `--data-dir` (default `~/.cheta`), `--camera webcam|reachy`.

### Reachy greeter

`examples/greeter.py` turns the robot's head to a recognised person and greets them by name,
mentioning something it remembers. It also runs on a laptop webcam.

```bash
python examples/greeter.py
python examples/greeter.py --camera reachy
```

## How it works

| Piece | Implementation |
| --- | --- |
| Face | OpenCV YuNet detector + SFace embedding |
| Voice | Silero VAD + SpeechBrain ECAPA embedding |
| Memory | BGE-small embeddings, searched with sqlite-vec |
| Storage | SQLite; face/voice embeddings and memory text Fernet-encrypted; key in the OS keyring (5 s timeout) or a 0600 key file |
| Matching | Cosine similarity with a configurable threshold; below it the answer is `None` |
| Fusion | Face and voice agreeing raises confidence; a single signal is capped at 0.8; a conflict returns `None` |

## Limits you should know about

- **Liveness is a head-turn check only.** It stops a flat photo held up to the camera. It does not
  stop video replay, masks or deepfakes. Do not use cheta to guard anything valuable.
- Memory text and face/voice embeddings are encrypted at rest. The *embedding of each memory* (used for
  search) is not, because SQLite's vector search must read it. It reveals roughly what a memory is
  about, not its text.
- `Person.confidence` is rescaled: a single signal tops out at 0.8.
- Voice recognition is weaker than face recognition; tune `voice_threshold` for your setting.
- Model revisions from the Hugging Face Hub are not pinned.
- Tested against the Reachy Mini SDK source and fakes, not yet on a physical robot.

## Development

```bash
uv sync
uv run pytest                  # offline tests, fake embeddings
uv run pytest -m network       # also downloads real models
uv run ruff check . && uv run ruff format --check .
uv run mypy cheta
```

See [CHANGELOG.md](CHANGELOG.md).
