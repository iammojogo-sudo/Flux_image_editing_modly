# FLUX.1 Kontext [dev] Image Edit — Modly Extension

Instruction-based image editing with Black Forest Labs' **FLUX.1 Kontext [dev]**. Give it one image and a text instruction and it returns edited image(s) — great for getting different views/angles of the same object, relighting, style changes, object/character edits, and successive refinements with low drift.

12B model. Runs on consumer GPUs through 4-bit and CPU-offload modes (see Memory Mode below).

---

## Before you install: this is a gated model

The weights require accepting a license on HuggingFace, so you must do this once before the **Download** step will work:

1. Open https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev and click **Agree** to accept the license.
2. Authenticate so Modly can pull the weights, either:
   - run `huggingface-cli login` and paste a token from https://huggingface.co/settings/tokens, **or**
   - set an `HF_TOKEN` environment variable to that token.

If the Download fails with an access/gated error, one of these two steps is missing.

---

## Installation

1. Open Modly → **Extensions** tab.
2. **Install from GitHub** and paste this repo URL.
3. Wait for setup to finish (installs PyTorch + diffusers + bitsandbytes into an isolated venv).
4. Click **Download** on the Edit Image node to fetch the weights from HuggingFace (only the diffusers components are pulled, ~24GB).

---

## Usage (Workflows tab)

1. Drag an **Image** node onto the canvas and point it at the image you want to edit.
2. Drag a **Text** node and type your edit instruction (e.g. `show the back of this object`, `rotate the camera 45 degrees to the left`, `three-quarter side view`).
3. Drag an **Edit Image** node. Connect the **Image** node into its image input and the **Text** node into its text input.
4. Connect the **Edit Image** output into your **Preview Image** node.
5. Hit **Run**.

With **Number of Outputs** = 1 the node outputs a single image (same as the text-to-image node). Set it to 2–4 to get multiple variations in one run (each uses a different seed); the node then outputs a list of images.

### Prompt tips
Kontext follows explicit, literal instructions best. Describe the change directly ("turn the car to face left", "view from above"), keep the rest of the scene implied as unchanged, and make one change at a time for the cleanest results.

---

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| Memory Mode | Auto | How the model is loaded (see below) |
| Steps | 28 | Higher = slower, a bit more detail |
| Guidance Scale | 2.5 | How strictly it follows the instruction (~2.5–4 works well) |
| Number of Outputs | 1 | 1–4 variations per run |
| Seed | 0 | 0 = random each run; a fixed value is reproducible |

### Memory Mode

| Mode | What it does | Rough VRAM |
|---|---|---|
| Auto | Picks 4-bit if the GPU has < 20GB, otherwise bf16 + offload | — |
| Low VRAM | 4-bit (nf4) transformer + T5, model CPU offload | ~10–12GB |
| Balanced | bf16 weights with model CPU offload | ~24GB |
| Max Speed | bf16 fully on GPU, no offloading (fastest) | ~24GB+ |

Low VRAM needs `bitsandbytes` (installed during setup). If it didn't install, that mode falls back to Balanced automatically.

---

## Notes

- First run is slower while weights load; later runs in the same session are faster.
- FLUX.1 Kontext [dev] is released under the **FLUX.1 [dev] Non-Commercial License** — review it for your use case. BFL also recommends output content filtering for any redistribution/deployment.
- Lighter alternative worth knowing about: **FLUX.2 [Klein]** is a newer unified generate/edit model that targets ~13GB VRAM. If you want a build around that instead, it slots into the same node structure with a different pipeline.
