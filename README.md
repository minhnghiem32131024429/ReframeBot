# ReframeBot

ReframeBot is a CBT-oriented chatbot for supporting university students with academic stress. It combines a fine-tuned Llama 3.1 model with a guardrail router (TASK_1/TASK_2/TASK_3) and optional RAG grounding from a CBT knowledge base.

## Model Repositories

Our trained models are available on Hugging Face:

- **[SFT Adapter](https://huggingface.co/Nhatminh1234/ReframeBot-SFT-Llama3.1-8B)** - Supervised Fine-Tuning adapter for Llama 3.1 8B
- **[DPO Adapter](https://huggingface.co/Nhatminh1234/ReframeBot-DPO-Llama3.1-8B)** - Direct Preference Optimization adapter
- **[Guardrail Classifier](https://huggingface.co/Nhatminh1234/ReframeBot-Guardrail-DistilBERT)** - Task classifier for message routing (CBT/Crisis/Out-of-scope)

## Features
- Fine-tuned Llama 3.1 8B (DPO adapter)
- Guardrail routing with crisis detection and out-of-scope redirection
- Optional RAG grounding over a CBT knowledge base
- FastAPI backend and a lightweight static web UI

## Quick Start

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

4. Create a local config file (recommended):
   - Copy `.env.example` to `.env`
   - Update paths as needed (example):
```env
GUARDRAIL_PATH=D:\\Work\\AI\\guardrail_model_retrained\\best
GUARDRAIL_CONTEXT_TURNS=3
```
   Notes:
   - `.env` is ignored by git by design.
   - Model checkpoints are not committed to this repo; point `GUARDRAIL_PATH` to your local folder.

5. Run the FastAPI server:
```bash
python app.py
```

6. Serve the web UI (in a new terminal):
```bash
cd web
python -m http.server 8080
```
   - Open: http://localhost:8080/

The UI will call the backend at `http://127.0.0.1:8000/chat`.

## Project Structure

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

## UI

- Glassmorphism-style layout (HTML/CSS)
- Responsive chat UI

## Configuration

### Change API URL
Edit `web/script.js`:
```javascript
const API_URL = "http://your-domain.com/chat";
```

### Guardrail settings
You can override routing behavior without editing code via `.env`:
- `GUARDRAIL_PATH`: local folder path to a transformers text-classification checkpoint
- `GUARDRAIL_CONTEXT_TURNS`: number of recent user turns concatenated for guardrail input (default: 3)
- `GUARDRAIL_CONTEXT_MAX_CHARS`: max characters used for guardrail input (default: 700)
- `CRISIS_SEMANTIC_SIM_THRESHOLD`, `CRISIS_SEMANTIC_SIM_MARGIN`: semantic crisis detector thresholds
- `ROUTER_EMBED_MODEL`: embedding model used for routing/RAG (default: `all-MiniLM-L6-v2`)

### Customize Colors
Edit `web/style.css` to change color scheme, glass effects, and more.

## Training

See `train.ipynb` for the complete training pipeline:
1. **SFT (Supervised Fine-Tuning)** - Base model adaptation
2. **DPO (Direct Preference Optimization)** - Response quality improvement
3. **Guardrail Training** - Task classification model

Optional scripts:
- `scripts/prepare_guardrail_data.py`: merge + deduplicate guardrail data (and add hard negatives)
- `scripts/train_guardrail.py`: retrain the guardrail classifier from a JSONL dataset

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Author

**Nghiem Nhat Minh**
- GitHub: [@minhnghiem32131024429](https://github.com/minhnghiem32131024429)
- Hugging Face: [@Nhatminh1234](https://huggingface.co/Nhatminh1234)

## Acknowledgments

- Meta AI for Llama 3.1
- Hugging Face for transformers and PEFT libraries
- FastAPI team
