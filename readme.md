
# 📄 Chat-with-PDF (Streamlit + Sentence-Transformers + OpenAI)

Ask questions about any text-based PDF and get answers grounded in its content.  
Powered by a lightweight Retrieval-Augmented Generation (RAG) stack:

- ✅ Streamlit 1.34+ with native chat UI
- ✅ Sentence-Transformers for embedding
- ✅ NumPy + scikit-learn for similarity
- ✅ OpenAI GPT-3.5-turbo for response generation

---

## 🧪 Demo

```bash
streamlit run app.py

📁 Project Structure

.
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .env                    # OpenAI API key (you create this)
├── .streamlit/
│   └── config.toml         # (Optional) silence PyTorch watcher warnings
└── README.md


🚀 Getting Started
1. Clone and install dependencies

git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf
conda create -n pdfchat python=3.10 -y
conda activate pdfchat
pip install -r requirements.txt

2. Set up your API key
Create a file called .env:

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

3. Run the app

streamlit run app.py

