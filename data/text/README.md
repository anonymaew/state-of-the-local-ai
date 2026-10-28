# Text inputs

`benchmarks/text/` needs one plain-text corpus for perplexity. The bytes are **not** committed:
the corpus is a Wikipedia extract and carries its own license.

## Perplexity corpus

- Source: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test`
  ([dataset revision `b08601e`](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3)).
- Local file used for the published PPL rows: `wiki.test.raw`, 1,292,013 bytes, 7,248 lines,
  sha256 `aca2f46735043bcfd0a44eca981d04627b9cdf74c4c9a04bf0856d04066f58fc`.
- Fetch and flatten it with the pinned revision:

```bash
.venv/bin/python - <<'PY'
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                  split="test", revision="b08601e04326c79dfdd32d625aee71d232d685c3")
with open("data/text/wiki.test.raw", "w", encoding="utf-8") as fh:
    fh.write("\n".join(ds["text"]))
PY
```

`ppl_mlx.py` reads `[chunks] × [ctx]` tokens (default 12 × 2048). If the regenerated file differs from
the sha256 above, treat the PPL as a new series: record the hash in the row's `label` and do not compare
it with the published numbers.
