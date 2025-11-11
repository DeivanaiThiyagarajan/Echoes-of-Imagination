# 🎨 Story to Image Generator - Performance Guide

## ⚡ Performance Comparison

### **On GPU (NVIDIA/AMD)**
| Metric | Stable Diffusion v1.5 | Latent Diffusion |
|--------|----------------------|-----------------|
| Model Used | ✅ (Best quality) | N/A |
| Parameters | 860M | - |
| Image Resolution | 512×512 | 256×256 |
| Time per Image | 15-30 seconds | N/A |
| Quality | Excellent | - |
| Memory | 6-8GB | 2-3GB |

### **On CPU (No GPU)**
| Metric | Stable Diffusion v1.5 | Latent Diffusion |
|--------|----------------------|-----------------|
| Model Used | ❌ (Too slow) | ✅ (Lightweight) |
| Parameters | 860M | 200M |
| Image Resolution | 512×512 | 256×256 |
| Time per Image | 30-60 minutes | 5-15 minutes |
| Quality | Excellent | Good |
| Memory | 6-8GB | 2-3GB |

---

## 🚀 How to Run

### **With GPU (Recommended)**
```bash
cd c:\Users\Aishu\Downloads\AML-2\Echoes-of-Imagination\ui
python just_ui.py
```
✅ Uses Stable Diffusion v1.5 (best quality)  
✅ 15-30 seconds per image  
✅ Best user experience  

### **With CPU Only**
```bash
cd c:\Users\Aishu\Downloads\AML-2\Echoes-of-Imagination\ui
python just_ui.py
```
⚠️ Uses Latent Diffusion (lightweight)  
⚠️ 5-15 minutes per image  
⚠️ Consider: Run overnight for story sequences  

---

## 📋 System Requirements

### **For GPU (Recommended)**
- NVIDIA GPU with 6GB+ VRAM (or AMD equivalent)
- CUDA 11.8+ installed
- PyTorch with GPU support
- **Expected runtime: 2-5 minutes for a 5-segment story**

### **For CPU Only**
- CPU with 8+ cores
- 8GB+ system RAM
- Python 3.9+
- **Expected runtime: 30-75 minutes for a 5-segment story**

---

## 🎯 What Changed

Your `just_ui.py` now:

1. **Auto-detects device** (GPU vs CPU)
2. **Selects appropriate model**:
   - GPU: Uses full Stable Diffusion v1.5
   - CPU: Uses lightweight Latent Diffusion
3. **Adjusts parameters dynamically**:
   - GPU: 512×512 resolution, 30 inference steps
   - CPU: 256×256 resolution, 50 inference steps
4. **Warns users about speed** if running on CPU

---

## 💡 Tips for Best Performance

### **If using GPU:**
- Close other applications to free VRAM
- Use 2-4 sentence story segments
- Adjust `inference_steps` down to 20 for faster generation

### **If using CPU:**
- Run with smaller story segments
- Use short, simple prompts
- Consider using scheduled/overnight runs
- Or: Install GPU support for better experience

---

## 🔧 To Force GPU Usage (if available)

Edit `just_ui.py` and uncomment this at the top:
```python
# Force GPU (will error if not available)
device = torch.device("cuda")
```

---

## ✅ Checklist

- [x] Auto-detect GPU/CPU
- [x] Use appropriate model for device
- [x] Dynamic image resolution (512 GPU, 256 CPU)
- [x] User-friendly warnings
- [x] Graceful fallback to CPU

**Status: Ready to use! 🚀**
