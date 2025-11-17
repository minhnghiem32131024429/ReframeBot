# Setup Guide

## Model Configuration

### LoRA Adapter Setup

1. **Download the adapter** from HuggingFace:
   - [ReframeBot-Llama-3.1-8B-Adapter](https://huggingface.co/Nhatminh1234/ReframeBot-Llama-3.1-8B-Adapter)

2. **Place the adapter** in your project directory. You can organize it as:
   ```
   reframebot-glassmorphism/
   ├── models/                    # Create this folder
   │   └── checkpoint-90/        # Your downloaded adapter
   ```

3. **Update the adapter path** in `app.py`:
   ```python
   # Change this line (around line 17):
   adapter_path = r"D:\Work\AI\results_reframebot_DPO\checkpoint-90"
   
   # To your actual path, for example:
   adapter_path = "./models/checkpoint-90"
   # or absolute path:
   adapter_path = r"C:\path\to\your\reframebot-glassmorphism\models\checkpoint-90"
   ```

### GPU Requirements

- **Minimum**: 8GB VRAM (with 4-bit quantization)
- **Recommended**: 12GB+ VRAM for better performance
- **CPU-only**: Not recommended (very slow inference)

### Testing the Server

After starting the server with `python app.py`, test it:

```bash
# Test API endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"I'm worried about the upcoming exam","history":[]}'
```

## Web Interface Setup

1. Make sure the FastAPI server is running on port 8000
2. Open `web/index.html` in your browser
3. Or use a local HTTP server:
   ```bash
   python -m http.server 8080
   # Then navigate to http://localhost:8080/web/
   ```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Use 4-bit quantization (already configured)
- Close other GPU applications

### Model Loading Errors
- Check adapter path is correct
- Ensure you have internet connection for base model download
- Verify HuggingFace token if model requires authentication

### CORS Errors
- Server is configured to allow all origins
- Make sure FastAPI server is running before opening web interface
- Check browser console for specific errors
