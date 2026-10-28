# Engine source

The engine source is a build dependency, not part of this repo.

Pinned: llama.cpp **v0.7.0** / commit `620d2225cc436ab0b001e3200e590e3b286ef9ab` (2026-10-15).
M5 needs build ≥ `b10734` for the Metal-4 tensor API and ≥ v0.7.0 for MTP; nixpkgs 0.6.0 has neither.

```bash
git clone --depth 1 --branch v0.7.0 https://github.com/ggml-org/llama.cpp src/llama.cpp
cmake -B src/llama.cpp/build -S src/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
cmake --build src/llama.cpp/build -j"$(sysctl -n hw.ncpu)" --config Release \
  --target llama-bench llama-cli llama-server llama-perplexity
```

Harnesses look for `src/llama.cpp/build/bin/llama-server`; no environment variable is needed.

## Optional: Kev

The decision track's `breadth-v1` partitions live in Kev's repo. Their public mirror is private for
licensing reasons, so the loader reads the clone first:

```bash
git clone https://github.com/jaredpalmer/kev src/kev
```

Nothing else in this repo requires `src/kev`.
