# DeepSeek PDF Assistant

This project is a Streamlit-based interactive PDF question-answering assistant powered by LangChain, FAISS, and Ollama's DeepSeek LLM. The application allows users to upload PDF documents, process them into semantic chunks, and retrieve accurate, context-aware answers.

---

## Requirements

- Python 3.8 or higher
- Streamlit
- LangChain
- Ollama CLI
- DeepSeek Model (DeepSeek-r1)

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/akshayks13/DeepSeek-PDF-Assistant.git
cd DeepSeek-PDF-Assistant
```

### Step 2: Set Up Virtual Environment (Optional)
```bash
python3 -m venv venv
source venv/bin/activate  # For Mac/Linux
.\venv\Scripts\activate  # For Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama

## **[👉 Follow the Ollama Installation Guide](https://ollama.com/)**  
Alternatively, use the command below:
```bash
brew install ollama
```

### Step 5: Download the DeepSeek Model

Ollama provides different variants of the DeepSeek model to suit various computational needs. Start by downloading the default 7B model or choose a variant that fits your requirements.

#### To Download the Default 7B Model:
```bash
ollama pull deepseek-r1
```

#### To Download Other Model Variants:
- **1.5B Model:**
  ```bash
  ollama pull deepseek-r1:1.5b
  ```
  Suitable for lightweight applications where memory is constrained.

- **14B Model:**
  ```bash
  ollama pull deepseek-r1:14b
  ```
  Balances reasoning power and resource usage.

- **70B Model:**
  ```bash
  ollama pull deepseek-r1:70b
  ```
  Offers advanced reasoning but requires significant RAM.

- **671B Model:**
  ```bash
  ollama pull deepseek-r1:671b
  ```
  Best for heavy computational tasks requiring superior reasoning.

> [!TIP]
> Choose a model variant based on your system's RAM and performance requirements.

---

## Running the Application

1. Start the Streamlit application.

   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL

   ![image](https://github.com/user-attachments/assets/e968c29c-3340-4f1c-97ca-461440b4921f)

---

## Usage

1. Upload a PDF file through the UI.
2. Enter your question in the provided input box.
3. View the response based on the uploaded document.

---

## Notes
- Ensure you have the DeepSeek model (`deepseek-r1`) downloaded before running the app.

---

## Acknowledgements
Special thanks to LangChain, FAISS, and Ollama for their amazing tools and libraries.
