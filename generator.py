import os
import random
import sys
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

from services.generators.base import BaseGenerator, smooth_progress

# keep stdout clean for the runner protocol
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)

HF_REPO = "black-forest-labs/FLUX.1-Kontext-dev"

# FluxKontextPipeline only needs the diffusers-format components. Pulling just
# these avoids also downloading the redundant ~24GB single-file checkpoint that
# ships in the same repo, roughly halving the download.
ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "transformer/*",
    "vae/*",
]


def _int(val, default):
    try:
        return int(val)
    except:
        return default


def _float(val, default):
    try:
        return float(val)
    except:
        return default


class FluxKontextEditGenerator(BaseGenerator):
    MODEL_ID     = "flux_kontext_dev_edit"
    DISPLAY_NAME = "FLUX.1 Kontext [dev] Image Edit"
    VRAM_GB      = 12

    def is_downloaded(self):
        check = self.download_check
        if check:
            return (self.model_dir / check).exists()
        return (self.model_dir / "model_index.json").exists()

    # ------------------------------------------------------------------ loading
    def load(self):
        mode = getattr(self, "_mem_mode", None) or "auto"

        if self._model is not None and getattr(self, "_loaded_mode", None) == mode:
            return
        if self._model is not None:
            self.unload()

        if not self.is_downloaded():
            self._download_weights()

        import torch
        from diffusers import FluxKontextPipeline

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        resolved = self._resolve_mode(mode)
        print("[FluxKontext] loading (%s) from %s" % (resolved, self.model_dir))

        pipe = None
        if resolved == "low_vram":
            pipe = self._load_quantized(torch)
            if pipe is None:
                print("[FluxKontext] 4-bit load unavailable, falling back to bf16 + offload")
                resolved = "balanced"

        if pipe is None:
            pipe = FluxKontextPipeline.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
                torch_dtype=self._dtype,
            )

        if self._device == "cuda":
            if resolved == "max_speed":
                pipe.to("cuda")
            else:  # balanced + low_vram both use model cpu offload
                pipe.enable_model_cpu_offload()
            try:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
            except Exception:
                pass
        else:
            print("[FluxKontext] CUDA not available — running on CPU, this will be extremely slow")
            pipe.to("cpu")

        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self._model = pipe
        self._loaded_mode = mode
        print("[FluxKontext] ready on %s" % self._device)

    def _resolve_mode(self, mode):
        mode = (mode or "auto").strip().lower()
        if mode != "auto":
            return mode
        try:
            import torch
            if not torch.cuda.is_available():
                return "balanced"
            _free_b, total_b = torch.cuda.mem_get_info()
            total_gb = total_b / (1024 ** 3)
            return "low_vram" if total_gb < 20 else "balanced"
        except Exception:
            return "balanced"

    def _load_quantized(self, torch):
        # nf4 4-bit transformer + T5 so the 12B model fits ~10-12GB with model offload
        try:
            from diffusers import FluxKontextPipeline, FluxTransformer2DModel
            from diffusers import BitsAndBytesConfig as DiffBnb
            from transformers import T5EncoderModel
            from transformers import BitsAndBytesConfig as TfBnb
        except Exception as e:
            print("[FluxKontext] 4-bit imports failed (%s)" % e)
            return None

        try:
            d_cfg = DiffBnb(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                str(self.model_dir),
                subfolder="transformer",
                quantization_config=d_cfg,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )

            t_cfg = TfBnb(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                str(self.model_dir),
                subfolder="text_encoder_2",
                quantization_config=t_cfg,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )

            pipe = FluxKontextPipeline.from_pretrained(
                str(self.model_dir),
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            return pipe
        except Exception as e:
            print("[FluxKontext] 4-bit load failed (%s)" % e)
            return None

    def unload(self):
        self._model = None
        self._device = None
        self._dtype = None
        self._loaded_mode = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------------------------------------------------------- generation
    def _to_pil(self, image_in):
        from PIL import Image
        if image_in is None:
            raise ValueError("no input image — connect an Image node to the Edit node")
        if isinstance(image_in, Image.Image):
            return image_in.convert("RGB")
        if isinstance(image_in, (bytes, bytearray)):
            return Image.open(BytesIO(bytes(image_in))).convert("RGB")
        if hasattr(image_in, "__fspath__") or isinstance(image_in, str):
            return Image.open(image_in).convert("RGB")
        if hasattr(image_in, "read"):
            return Image.open(image_in).convert("RGB")
        raise ValueError("unrecognized image input type: %r" % type(image_in))

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        import torch

        params = params or {}
        self._mem_mode = params.get("memory_mode") or "auto"

        if self._model is None or getattr(self, "_loaded_mode", None) != self._mem_mode:
            self.load()

        prompt = (params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("no edit instruction — connect a Text node to the Edit node")

        image = self._to_pil(image_bytes)

        steps     = _int(params.get("steps"), 28)
        cfg       = _float(params.get("guidance_scale"), 2.5)
        n_images  = max(1, min(_int(params.get("num_images"), 1), 4))
        base_seed = _int(params.get("seed"), 0)

        self._report(progress_cb, 5, "starting up")
        self._check_cancelled(cancel_event)

        if self.outputs_dir:
            out_dir = self.outputs_dir
        else:
            out_dir = self.model_dir.parent.parent.parent / "outputs" / self.MODEL_ID
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for i in range(n_images):
            self._check_cancelled(cancel_event)

            if base_seed == 0:
                seed = random.randint(1, 2**31 - 1)
            else:
                seed = base_seed + i
            # FLUX uses a CPU generator even on CUDA (matches the diffusers examples
            # and avoids device-mismatch under cpu offload).
            gen = torch.Generator(device="cpu").manual_seed(seed)

            lo = 10 + int(85 * (i / float(n_images)))
            hi = 10 + int(85 * ((i + 1) / float(n_images)))
            label = "editing" if n_images == 1 else "editing %d/%d" % (i + 1, n_images)
            self._report(progress_cb, lo, label)

            stop = threading.Event()
            ticker = None
            if progress_cb:
                ticker = threading.Thread(
                    target=smooth_progress,
                    args=(progress_cb, lo, hi, label, stop),
                    daemon=True,
                )
                ticker.start()

            try:
                with torch.inference_mode():
                    result = self._model(
                        image=image,
                        prompt=prompt,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=gen,
                    )
                out_img = result.images[0]
            finally:
                stop.set()
                if ticker:
                    ticker.join(timeout=1.0)

            filename = "flux_kontext_%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8])
            out_path = out_dir / filename
            out_img.save(str(out_path), format="PNG")
            paths.append(str(out_path))
            print("[FluxKontext] saved %s (seed %d)" % (out_path, seed))

        self._report(progress_cb, 100, "done")

        # single output -> return a path string (same contract as the t2i node so the
        # existing Preview node works unchanged); multiple -> return a list of paths.
        if len(paths) == 1:
            return paths[0]
        return paths

    # ----------------------------------------------------------------- download
    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download

        repo = self.hf_repo or HF_REPO
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or None
        )

        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[FluxKontext] downloading diffusers components from %s" % repo)

        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(self.model_dir),
                allow_patterns=ALLOW_PATTERNS,
                token=token,
            )
        except Exception as e:
            msg = str(e).lower()
            gated = any(k in msg for k in (
                "gated", "401", "403", "authoriz", "access to model",
                "awaiting", "must agree", "accept the license", "restricted",
            ))
            if gated:
                raise RuntimeError(
                    "FLUX.1 Kontext [dev] is a gated model. "
                    "1) Accept the license at https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev  "
                    "2) run 'huggingface-cli login' or set the HF_TOKEN environment variable, "
                    "then download again."
                )
            raise

        print("[FluxKontext] download complete")
