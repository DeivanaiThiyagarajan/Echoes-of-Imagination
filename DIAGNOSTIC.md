# Quick Diagnostic Code - Run This First

Copy and paste this into a new cell to diagnose the error:

```python
# === DIAGNOSTIC CELL ===
import traceback
import sys

print("="*60)
print("DIAGNOSTIC: Testing Model Components")
print("="*60)

try:
    print("\n1. Testing tokenizer...")
    print(f"   Tokenizer type: {type(tokenizer)}")
    print(f"   Model max length: {tokenizer.model_max_length}")
    
    # Test tokenization
    test_caption = "a photo of a cat"
    tokens = tokenizer(test_caption, padding="max_length", max_length=77, 
                       truncation=True, return_tensors="pt")
    print(f"   ✓ Tokenization works. Shape: {tokens.input_ids.shape}")
    
except Exception as e:
    print(f"   ✗ Tokenizer error: {e}")
    traceback.print_exc()

try:
    print("\n2. Testing text encoder...")
    print(f"   Encoder type: {type(text_encoder)}")
    
    # Test encoding
    with torch.no_grad():
        output = text_encoder(tokens.input_ids.to(DEVICE))
    print(f"   ✓ Text encoder works. Output shape: {output[0].shape}")
    
except Exception as e:
    print(f"   ✗ Text encoder error: {e}")
    traceback.print_exc()

try:
    print("\n3. Testing VAE...")
    print(f"   VAE type: {type(vae)}")
    
    # Create dummy image
    dummy_image = torch.randn(1, 3, 256, 256).to(DEVICE)
    
    with torch.no_grad():
        latents = vae.encode(dummy_image).latent_dist.sample()
    print(f"   ✓ VAE works. Latent shape: {latents.shape}")
    
except Exception as e:
    print(f"   ✗ VAE error: {e}")
    traceback.print_exc()

try:
    print("\n4. Testing UNet...")
    print(f"   UNet type: {type(unet)}")
    
    # Create dummy inputs
    timesteps = torch.tensor([500]).to(DEVICE)
    
    with torch.no_grad():
        pred = unet(latents, timesteps, output[0]).sample
    print(f"   ✓ UNet works. Output shape: {pred.shape}")
    
except Exception as e:
    print(f"   ✗ UNet error: {e}")
    traceback.print_exc()

try:
    print("\n5. Testing dataloader wrapper...")
    print(f"   Wrapper type: {type(wrapped_train_dataloader)}")
    print(f"   Total batches: {len(wrapped_train_dataloader)}")
    
    # Get first batch
    batch = next(iter(wrapped_train_dataloader))
    print(f"   ✓ First batch loaded")
    print(f"     - Latent shape: {batch['latent'].shape}")
    print(f"     - Encoder states shape: {batch['encoder_hidden_states'].shape}")
    
except Exception as e:
    print(f"   ✗ Dataloader error: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("END DIAGNOSTIC")
print("="*60)
```

---

## What to do:

1. **Create a new cell** in your notebook
2. **Copy the diagnostic code above** into it
3. **Run it** 
4. **Share the output** - it will tell you exactly which component is failing

Then I can give you the exact fix!

**Common errors it might find:**
- Tokenizer not compatible with text encoder
- Text encoder output shape mismatch
- VAE encoding issues
- UNet input shape mismatch
- Dataloader wrapper failure

Let me know what the diagnostic shows! 🔍
