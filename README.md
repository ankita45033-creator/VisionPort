# VisionPort

A GPU-trained, GPU-served vision system for Indian traffic.

## Current version
v0.1 - pretrained YOLOv8n demo, FastAPI gateway, request trace

## Codebase tour
| Part | Location | Will contain |
|---|---|---|
| Data pipeline | data/, train/dataset.py | Data loading and preprocessing |
| Training loop | train/train.py | Model training |
| Architecture file | train/model.py | Model architecture |
| Evaluation harness | eval/ | Model evaluation |
| Optimisation script | optimise/ | Model optimisation |
| Triton config | serve/triton/ | Model serving configuration |

## Environment
See env_report.md. Recreate with: pip install -r requirements.txt

## How to run
uvicorn serve.gateway_trace:app --reload   (then open /docs)

## Known limitations
- Pretrained on COCO, not on Indian traffic. No auto-rickshaw class.
