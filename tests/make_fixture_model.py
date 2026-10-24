"""Build the tiny random-init BPE model the web tests load (web/test/fixtures/bpe_model{,_fp16}/).

Usage: python tests/make_fixture_model.py
The weights are untrained (seeded random init), so generated text is gibberish; the fixture only exercises the
loading, tokenization, streaming and stop-string paths of the web handler. Re-running gives identical files."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import export_weights  # noqa: E402
import train  # noqa: E402
from bpe import BPE  # noqa: E402

CFG = {"block": 16, "n_layer": 2, "n_head": 2, "n_embd": 32, "dropout": 0.0}
VOCAB = 300
SEED = 1234
SAMPLE = """User:
Hello, how are you today?

Assistant:
I am doing well, thank you for asking. The weather is lovely and the garden is full of flowers.

User:
What should I cook for dinner tonight?

Assistant:
You could make a simple pasta with tomatoes, garlic and fresh basil. It takes about twenty minutes.

User:
Tell me something about the ocean.

Assistant:
The ocean covers more than seventy percent of the planet. Whales, dolphins and many small fish live there,
and the deepest point is nearly eleven kilometers below the surface.

User:
Can you write a short poem about the moon?

Assistant:
The silver moon climbs over the hill,
the night is quiet, the river is still.
She watches the town as the people sleep,
and keeps the secrets the shadows keep.

User:
Why is the sky blue?

Assistant:
Sunlight is made of many colors. Blue light is scattered more strongly by the air than red light,
so when you look up during the day, the sky appears blue.

User:
Thanks for the help!

Assistant:
You are welcome. Have a wonderful evening, and come back whenever you have another question.
"""


def main():
    torch.manual_seed(SEED)
    tok = BPE.train(SAMPLE, VOCAB)
    model = train.GPT(CFG, len(tok.vocab))
    ckpt = {"model": model.state_dict(), "chars": None, "tokenizer": train.tokenizer_dict(tok),
            "config": CFG, "step": 0}
    out = ROOT / "web" / "test" / "fixtures"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fixture.pt")
        torch.save(ckpt, path)
        export_weights.main(["--ckpt", path, "--out", str(out / "bpe_model")])
        export_weights.main(["--ckpt", path, "--out", str(out / "bpe_model_fp16"), "--fp16"])
    print(f"vocab {len(tok.vocab)} ({len(tok.alphabet)} characters + {len(tok.merges)} merges)")


if __name__ == "__main__":
    main()
