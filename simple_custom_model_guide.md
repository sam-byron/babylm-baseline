# Simple Custom Model Guide for lm_eval

## The Official Way

According to HuggingFace documentation and lm_eval source code, you just need to:

1. **Save your model properly with the required files**
2. **Use `trust_remote_code=True` when loading**

That's it! No complex registration, no weight copying, no special scripts.

## Required Files in Model Directory

Your model directory needs these files:
- `config.json` - Model configuration 
- `pytorch_model.bin` or `model.safetensors` - Model weights
- `modeling_*.py` - Your model code (e.g., `modeling_ltg_bert.py`)
- `configuration_*.py` - Your config code (e.g., `configuration_ltg_bert.py`)
- Tokenizer files: `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, etc.

## Key Requirements

1. **Model Type**: Your config must have a unique `model_type` (e.g., `"ltg_bert"`)

2. **Auto Registration**: Your config and model classes need `register_for_auto_class()`:
   ```python
   # In your model saving script:
   config.register_for_auto_class()
   model.register_for_auto_class("AutoModelForSequenceClassification")
   ```

3. **File Naming**: Use the pattern `modeling_{model_type}.py` and `configuration_{model_type}.py`

## Usage with lm_eval

```bash
lm_eval --model hf \
    --model_args pretrained=/path/to/your/model,trust_remote_code=True \
    --tasks your_tasks
```

## What Was Wrong With Our Complex Approach

- HuggingFace already handles custom model loading
- `trust_remote_code=True` is the standard parameter for custom models
- No need for manual AutoModel registration
- No need for special loading scripts
- No need for weight copying between model types

## Next Steps

1. Update your model saving to use proper file names
2. Add `register_for_auto_class()` calls
3. Test with `trust_remote_code=True`
4. Use directly with lm_eval

This is the official, supported, simple way that works with all HuggingFace tools and evaluation frameworks.
