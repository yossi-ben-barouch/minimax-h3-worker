# MiniMax H3 Base Ref2VA Worker

RunPod Serverless worker for `MiniMaxAI/MiniMax-H3` `ref2va`: prompt plus one to nine ordered image references in, a video/audio MP4 plus poster image in the private Viralflix output bucket out.

The request-level `quality_profile` keeps the original Base path available and adds a pinned Ref2VA-specific 8-step Turbo path. The Turbo adapter is the native Diffusers-format `lightx2v/Minimax-h3-Turbo` 768p Ref2V checkpoint, pinned by repository revision and SHA-256; it is never substituted with an FL2VA or ComfyUI-pruned adapter.

## Why It Is Separate

Ref2VA uses its own transformer partition and the shared Qwen3-VL conditioner. The official Diffusers documentation reports a 61.7 GB transformer and a 62.1 GB conditioner, so this worker has a dedicated 200 GB RunPod Network Volume. The worker uses supported automatic CPU offload and targets a 96 GB RTX PRO 6000 Blackwell Server Edition or larger. It is intentionally configured with `workersMin: 0`.

## License Gate

The official MiniMax H3 Community License excludes the United States, European Union, United Kingdom, and South Korea from its applicable territory, including hosted use and outputs. Do not stage the checkpoint or activate this worker for Viralflix until written authorization or another production license covers the intended service territory. Commercial use also requires visible `MiniMax H3` attribution and user terms carrying the model's use restrictions.

## Deployment

1. Build and publish `ghcr.io/yossi-ben-barouch/minimax-h3-worker:<tag>`.
2. Create a 200 GB Network Volume in the endpoint's data center. After the endpoint is configured, submit one protected staging job so the model is downloaded directly to the mounted volume:

   ```json
   {
     "input": {
       "worker_token": "<same worker token>",
       "admin_action": "stage_models"
     }
   }
   ```

   Before a paid render, verify that every Ref2VA component loads from the
   staged volume without network fallback:

   ```json
   {
     "input": {
       "worker_token": "<same worker token>",
       "admin_action": "validate_components"
     }
   }
   ```

3. Create a RunPod Serverless template using the image, attach the volume to the endpoint, and configure:

   ```text
   MINIMAX_H3_WORKER_TOKEN=<random 32-byte hex>
   SUPABASE_URL=<project URL>
   SUPABASE_SERVICE_ROLE_KEY=<server-only key>
   OUTPUT_BUCKET=generation-outputs
   MODELS_DIR=/runpod-volume/models
   MINIMAX_H3_MODEL_DIR=/runpod-volume/models/minimax-h3
   MINIMAX_H3_GPU_MEMORY_RESERVE=32GB
   MINIMAX_H3_ATTENTION_BACKEND=_flash_3_hub
   MINIMAX_H3_TURBO_LORA_REPO=lightx2v/Minimax-h3-Turbo
   MINIMAX_H3_TURBO_LORA_REVISION=0eebcc7e79f9cb200927c80b8e7595265b770e34
   MINIMAX_H3_TURBO_LORA_FILENAME=minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors
   MINIMAX_H3_TURBO_LORA_SHA256=9bac880b1a5d7ac052171cf6cce769f0cceaaa42ffa51de4b8e41143a2bdd2d2
   ```

4. Configure the same endpoint id and worker token in Supabase Edge Function secrets. The `minimax_h3` adapter submits a job quickly; RunPod's webhook and the existing poller finalize the app job asynchronously.

## Worker Input

```json
{
  "worker_token": "server-only shared secret",
  "_job_id": "generation UUID",
  "_user_id": "Supabase user UUID",
  "prompt": "subject_definitions: ...",
  "reference_image_urls": ["https://signed-private-input.example/one.jpg"],
  "duration_seconds": 5,
  "width": 768,
  "height": 1344,
  "num_inference_steps": 30,
  "quality_profile": "base",
  "enable_audio": true
}
```

For the optimized profile, send `"quality_profile": "turbo_ref2va_8step"` and `"num_inference_steps": 8`. Profile selection is serialized with inference, so one worker can safely alternate Base and Turbo requests without cross-request adapter state.

The worker validates reference count, image MIME type/size, 5-15 second duration, and the H3 768px short-edge / 32px canvas invariants. It returns only storage paths and diagnostics, never video base64.
