# SmolLM-135M WDA (EMA)

This export contains WDA+gamma+backbone weights saved from `adapter_final_ema.pt`.

Files:
- `adapter_model.safetensors`: weights
- `adapter_config.json`: WDA hyperparams + base model id

To load, use `scripts/run_smol135m_wda_ema_inference.py` or adapt `export/.../load_wda.py`.
