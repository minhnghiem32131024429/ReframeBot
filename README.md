# ReframeBot 🤖💭

A chatbot interface with glassmorphism design, powered by Llama 3.1 finetuned model to help students reframe negative thoughts about academic stress using CBT techniques.

## 🤗 Model Repositories

Our trained models are available on Hugging Face:

- **[SFT Adapter](https://huggingface.co/Nhatminh1234/ReframeBot-SFT-Llama3.1-8B)** - Supervised Fine-Tuning adapter for Llama 3.1 8B
- **[DPO Adapter](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B)** - Direct Preference Optimization adapter
- **[Guardrail Classifier](https://huggingface.co/Nhatminh1234/ReframeBot-Guardrail-DistilBERT)** - Task classifier for message routing (CBT/Crisis/Out-of-scope)

## ✨ Features
- **AI-Powered Chat** - Finetuned Llama 3.1 8B with DPO optimization
- **Guardrail System** - Automatic task classification and crisis detection
- **Real-time Chat** - Fast and responsive messaging
- **Responsive Design** - Works on all devices
- **RESTful API** - FastAPI backend

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/minhnghiem32131024429/ReframeBot.git
cd ReframeBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download models from Hugging Face:
   - **SFT Adapter**: [ReframeBot-SFT-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-SFT-Llama3.1-8B)
   - **DPO Adapter**: [ReframeBot-DPO-Llama3.1-8B](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B)
   - **Guardrail Model**: [ReframeBot-Guardrail-DistilBERT](https://huggingface.co/Nhatminh1234/ReframeBot-Guardrail-DistilBERT)

4. Update model paths in `app.py`:
```python
ADAPTER_PATH = "./path/to/dpo-adapter"
GUARDRAIL_MODEL_PATH = "./path/to/guardrail-model"
```

5. Run the FastAPI server:
```bash
python app.py
```

6. Open the web interface:
   - Navigate to: http://localhost:8000
   - Or open `web/index.html` directly

## 📁 Project Structure

```
ReframeBot/
├── app.py                  # FastAPI backend with guardrail integration
├── train.ipynb             # Training notebook (SFT + DPO + Guardrail)
├── web/
│   ├── index.html          # Main HTML file
│   ├── style.css           # Glassmorphism styles
│   └── script.js           # Frontend logic
├── data/
│   ├── dataset.jsonl       # SFT training data
│   ├── dataset_dpo.jsonl   # DPO training data
│   └── guardrail_dataset.jsonl  # Guardrail training data
├── scripts/
│   ├── check_data.py       # Dataset validation
│   ├── check_dpo_dataset.py
│   ├── push_to_hub.py      # Upload single model
│   └── push_all_models.py  # Upload all models
├── docs/
│   └── SETUP.md            # Detailed setup guide
├── Utils/                  # Background assets
├── requirements.txt        # Python dependencies
└── README.md          
```

## 🎨 UI Features

- **Glassmorphism Design**: Frosted glass effect with backdrop blur
- **Custom Background**: Beautiful gradient or custom image support
- **Smooth Animations**: Floating bubbles and message animations
- **Hidden Scrollbars**: Clean, minimalist interface
- **Responsive Layout**: Adapts to all screen sizes

## ⚙️ Configuration

### Change API URL
Edit `web/script.js`:
```javascript
const API_URL = "http://your-domain.com/chat";
```

### Customize Colors
Edit `web/style.css` to change color scheme, glass effects, and more.

## 🧠 Training Pipeline

See `train.ipynb` for the complete training pipeline:
1. **SFT (Supervised Fine-Tuning)** - Base model adaptation
2. **DPO (Direct Preference Optimization)** - Response quality improvement
3. **Guardrail Training** - Task classification model

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 👤 Author

**Nghiem Nhat Minh**
- GitHub: [@minhnghiem32131024429](https://github.com/minhnghiem32131024429)
- Hugging Face: [@Nhatminh1234](https://huggingface.co/Nhatminh1234)

## 🙏 Acknowledgments

- Meta AI for Llama 3.1
- Hugging Face for transformers and PEFT libraries
- FastAPI team
