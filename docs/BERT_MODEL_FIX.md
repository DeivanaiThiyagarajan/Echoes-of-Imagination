# BERT Model Download Error - Fix Guide

## 🔴 Error You Got:
```
OSError: google/bert-base-uncased is not a local folder and is not a valid model identifier
```

## ✅ Why It Happened:
1. **Internet connection issue** - Can't reach HuggingFace Hub
2. **Wrong model name** - Used `google/bert-base-uncased` instead of `bert-base-uncased`
3. **Network firewall** - Corporate/institutional network blocking downloads
4. **Cache issues** - Corrupted local model cache

## 🛠️ Quick Fixes (Try in Order)

### Fix 1: Use Correct Model Name (RECOMMENDED)
The error was because of using the wrong namespace. Change this:
```python
tokenizer = BertTokenizer.from_pretrained("google/bert-base-uncased")
```

To this:
```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

✅ **I already fixed this in your notebook!**

---

### Fix 2: Check Internet Connection
Run this in a cell:
```python
import urllib.request

try:
    urllib.request.urlopen('https://huggingface.co', timeout=5)
    print("✓ Internet OK")
except:
    print("✗ Cannot reach HuggingFace")
```

If it fails → **Use Fix 4 (offline mode)**

---

### Fix 3: Clear and Retry Cache
```python
import shutil
from pathlib import Path

# Clear HuggingFace cache
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
print(f"Clearing cache: {cache_dir}")
shutil.rmtree(cache_dir, ignore_errors=True)

# Now model will re-download
print("✓ Cache cleared. Retry model loading.")
```

---

### Fix 4: Download Models Manually (Offline Mode)
If you have no internet or firewall issues:

**Option A: Download on different machine with internet**
```bash
pip install huggingface-hub
huggingface-cli download bert-base-uncased
huggingface-cli download CompVis/ldm-text2im-large-256
```
Then copy `~/.cache/huggingface/` to your machine.

**Option B: Use local model files**
```python
# If you have model files locally:
tokenizer = BertTokenizer.from_pretrained("/path/to/bert-base-uncased")
text_encoder = BertModel.from_pretrained("/path/to/bert-base-uncased")
```

---

### Fix 5: Use Hugging Face Token (for Private Models)
```bash
huggingface-cli login
# Enter your token when prompted
```

Then in Python:
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    use_auth_token=True  # Add this
)
```

---

## 📊 What I Fixed in Your Notebook:

### Cell 4 - Model Loading:
Added **automatic fallback** logic:
```python
if is_ldm:
    try:
        # Try LDM's built-in BERT
        tokenizer = BertTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
        text_encoder = BertModel.from_pretrained(config.model_id, subfolder="text_encoder")
    except Exception as e:
        # Fallback to standard BERT with CORRECT name
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_encoder = BertModel.from_pretrained("bert-base-uncased")
```

### New Cell - Diagnostics:
Added a **new cell** that checks:
- ✓ Internet connection
- ✓ HuggingFace cache location
- ✓ If model is already downloaded
- ✓ Disk space available

---

## 🚀 Quick Steps to Run Now:

1. **Run Cell 4** (Diagnostics) first
   - See if models are cached or need download
   - Check internet connection

2. **Run Cell 5** (Model Loading)
   - Should automatically use fallback if needed
   - Shows which BERT variant loaded

3. If still fails → See troubleshooting table below

---

## 🔍 Troubleshooting Table

| Issue | Check | Solution |
|-------|-------|----------|
| **Can't reach HuggingFace** | Run diagnostics cell | Use cached models or Fix 4 |
| **Still says wrong model name** | Look for `google/` prefix | Use `bert-base-uncased` |
| **Slow download** | Check disk space | Reduce batch size or skip heavy models |
| **Model 404 error** | Check model exists on HF | Use official model names |
| **Permission denied** | Check cache permissions | `chmod 755 ~/.cache/huggingface/` |

---

## 📝 Model Names Reference

### ✅ Correct Names:
- `bert-base-uncased` ← Use this!
- `CompVis/ldm-text2im-large-256`
- `runwayml/stable-diffusion-v1-5`

### ❌ Wrong Names (Don't Use):
- ~~`google/bert-base-uncased`~~ ← OLD (fixed!)
- ~~`bert/base/uncased`~~
- ~~`bert_base_uncased`~~

---

## 💾 Alternative: Use Smaller/Faster Models

If downloads are too slow, use lighter alternatives:

```python
# Instead of full BERT:
tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Advantages:
# - 40% smaller
# - 2× faster
# - Still good quality
```

---

## ✅ Your Notebook Already Handles This!

The updated Cell 4 includes:
- ✓ Correct model names
- ✓ Try/except fallback
- ✓ Diagnostic messages
- ✓ Support for both LDM and SD

**Just re-run the cells and it should work!** 🚀

---

## 📞 If Still Failing:

1. **Check error message** - Read the full error, not just first line
2. **Run diagnostics cell** - See what's cached
3. **Check internet** - Can you open huggingface.co in browser?
4. **Check disk space** - `df -h` on Linux/Mac or `dir` on Windows
5. **Try Fix 3** - Clear cache and retry

**Still stuck?** Share the full error message including:
- Device (CPU/GPU)
- Internet status (online/offline/behind proxy)
- Disk space available
- Full error traceback
