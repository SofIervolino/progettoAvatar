I passaggi sono pensati da svolgere in cmd, ricordarsi di modificare opportunamente le directory.

1. Si consiglia di creare un ambiente virtuale
python -m venv llama_env

Quindi aprirlo
source llama_env/bin/activate

2. Installare le dipendenze nell'abiente virtuale dal file requirements.txt
pip install -r requirements.txt

3. Scarica il modello Mistral-7B-Instruct-v0.2 da Hugging Face
huggingface-cli login

poi
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
tokenizer = AutoTokenizer.from_pretrained('Mistral-7B-Instruct-v0.2', use_auth_token=True); \
model = AutoModelForCausalLM.from_pretrained('Mistral-7B-Instruct-v0.2', use_auth_token=True)"

Questo scaricherà e salverà il modello nella tua cache locale (~/.cache/huggingface).

4. Prepara i dati per Chroma (indice semantico)
Estrare testo da un PDF (all.pdf), Dividerlo in chunk e Salvare in .txt in ./data/BigData
python estrazione.py

5. Costruisci l’indice Chroma
python chroma.py

6. Avvia il server FastAPI con GPU (Aggiungere le Api di Hugging Face per il modello e nel caso serva modificare device="cuda")
python server.py

7. Fare domande
Si consiglia di usare il file Test.ipynb per fare domande e ricevere risposte

