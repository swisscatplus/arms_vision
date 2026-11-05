"""Remove the background (the grey wall) of the camera photograph (panel A of fig:platform).

Input : paper/data/photos/2026-10-02_cameras_1.jpg (panel A of the compact fig_cameras layout)
Output: paper/data/photos/2026-10-02_cameras_1_nobg.png (RGBA, background transparent, as workbench_nobg.png)
        paper/data/photos/2026-10-02_cameras_1_nobg_mask.png (the foreground mask, 0/255)
        paper/data/photos/2026-10-02_cameras_1_nobg.json (prompt, scores, areas, tokenizer)

Method (for the caption's disclosure, Wiley image policy): SAM 3 (facebook/sam3, Sam3Model, Hugging Face
transformers) is given the text prompt PROMPT; the instances it returns above
THRESHOLD are the background. Everything else is kept, except disconnected parts smaller than MIN_PART of the
kept area (thin strips of wall along the photo's borders). The edge of the kept area is softened by a 1.5 px
Gaussian, which becomes the alpha channel. No pixel is altered; only the alpha channel is added.

The SAM 3 copy in the cache this runs from came without its tokenizer files (its tokenizer has 2 tokens, so every
text prompt arrives empty). The tokenizer is then rebuilt from the CLIP byte-pair vocabulary SAM 3 uses
(bpe_simple_vocab_16e6.txt.gz of the openai `clip` package, --bpe), written as vocab.json and merges.txt.

Run with the venv that has transformers (see segment_side_views.py), e.g.
    HF_HUB_CACHE=<cache> HF_HUB_OFFLINE=1 VENV/bin/python paper/analysis/tools/clean_photo_background_cameras.py \\
        [--device auto|mps|cpu] [--bpe .../site-packages/clip/bpe_simple_vocab_16e6.txt.gz]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label

ROOT = Path(__file__).resolve().parents[2]                     # paper/
SRC = ROOT / "data" / "photos" / "2026-10-02_cameras_1.jpg"
OUT = SRC.with_name(SRC.stem + "_nobg.png")
OUT_MASK = SRC.with_name(SRC.stem + "_nobg_mask.png")
OUT_JSON = SRC.with_name(SRC.stem + "_nobg.json")
REPO = "facebook/sam3"
BPE = Path("bpe_simple_vocab_16e6.txt.gz")   # from the openai `clip` package (site-packages/clip/); pass --bpe

PROMPT = "remove the grey background"
THRESHOLD = 0.3                         # instance score (query score x presence) for an instance to count
MIN_PART = 0.01                         # drop kept parts smaller than this fraction of the kept area
FEATHER_PX = 1.5


def bytes_to_unicode() -> dict[int, str]:
    """The byte-to-character table of the CLIP byte-pair tokenizer."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def clip_tokenizer(bpe: Path, folder: Path):
    """CLIP's tokenizer from its byte-pair vocabulary, as the openai `clip` SimpleTokenizer builds it."""
    import transformers
    merges = gzip.open(bpe).read().decode("utf-8").split("\n")[1:49152 - 256 - 2 + 1]
    vocab = list(bytes_to_unicode().values())
    vocab = vocab + [v + "</w>" for v in vocab] + ["".join(m.split()) for m in merges]
    vocab += ["<|startoftext|>", "<|endoftext|>"]
    (folder / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(vocab)}))
    (folder / "merges.txt").write_text("#version: 0.2\n" + "\n".join(merges) + "\n")
    (folder / "tokenizer_config.json").write_text(json.dumps(dict(
        tokenizer_class="CLIPTokenizer", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
        unk_token="<|endoftext|>", pad_token="<|endoftext|>", model_max_length=77, do_lower_case=True)))
    return transformers.AutoTokenizer.from_pretrained(folder)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bpe", type=Path, default=BPE)
    a = ap.parse_args()
    import torch
    import transformers
    dev = a.device if a.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    proc = transformers.Sam3Processor.from_pretrained(REPO)
    tokenizer_src = "the checkpoint's"
    with tempfile.TemporaryDirectory(prefix="clip_tokenizer_") as td:
        if getattr(proc.tokenizer, "vocab_size", 0) < 1000:          # the cache came without the tokenizer files
            proc.tokenizer = clip_tokenizer(a.bpe, Path(td))
            tokenizer_src = f"rebuilt from {a.bpe} (sha256 {hashlib.sha256(a.bpe.read_bytes()).hexdigest()[:16]}...)"
        tokens = proc.tokenizer.convert_ids_to_tokens(proc.tokenizer(PROMPT)["input_ids"])
        assert len(tokens) > 2, f"the prompt tokenizes to nothing: {tokens}"
        model = transformers.Sam3Model.from_pretrained(REPO).to(dev).eval()

        img = Image.open(SRC).convert("RGB")
        x = proc(images=img, text=PROMPT, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**x)
        res = proc.post_process_instance_segmentation(out, threshold=THRESHOLD, mask_threshold=0.5,
                                                      target_sizes=x["original_sizes"].tolist())[0]
    masks = res["masks"].cpu().numpy().astype(bool)
    scores = res["scores"].cpu().numpy()
    assert len(masks), f"SAM 3 found no instance of {PROMPT!r} above {THRESHOLD}"
    background = masks.any(axis=0)
    fg = ~background
    lab, n = label(fg)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes >= MIN_PART * fg.sum()) + 1
    fg = np.isin(lab, keep)                                             # strips of wall at the borders dropped

    alpha = (gaussian_filter(fg.astype(float), FEATHER_PX) * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(np.dstack([np.asarray(img), alpha]), "RGBA").save(OUT)
    Image.fromarray((fg * 255).astype(np.uint8)).save(OUT_MASK)
    rec = dict(model=REPO, model_class="Sam3Model", transformers=transformers.__version__, device=dev,
               prompt=PROMPT, prompt_tokens=tokens, tokenizer=tokenizer_src, threshold=THRESHOLD,
               presence=float(out.presence_logits.sigmoid().ravel()[0]),
               instance_scores=[float(s) for s in scores],
               background_fraction=float(background.mean()), kept_parts=int(len(keep)), dropped_parts=int(n - len(keep)),
               min_part_fraction=MIN_PART, foreground_fraction=float(fg.mean()), feather_px=FEATHER_PX)
    OUT_JSON.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
