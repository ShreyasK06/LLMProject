"""Every relative Markdown link and image in the docs points at a file that exists in the repo."""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ["README.md", "results/README.md", "docs/GPU_RUNBOOK.md"]
LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
FENCE = re.compile(r"^(```|~~~).*?^\1", re.S | re.M)


def relative_links(doc):
    text = open(os.path.join(ROOT, doc), encoding="utf-8").read()
    text = FENCE.sub("", text)  # links inside code blocks are not links
    for target in LINK.findall(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
            continue  # http(s), mailto and in-page anchors
        yield target.split("#", 1)[0]


@pytest.mark.parametrize("doc", DOCS)
def test_relative_links_resolve(doc):
    base = os.path.dirname(os.path.join(ROOT, doc))
    missing = [t for t in relative_links(doc) if not os.path.exists(os.path.normpath(os.path.join(base, t)))]
    assert not missing, f"{doc} links to missing files: {missing}"


def test_readme_states_origin():
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert "class project" in text
    assert "from scratch" in text
