## 安裝 Ollama 並拉取模型
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## 安裝 Python 依賴
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 啟動
streamlit run src/app.py
