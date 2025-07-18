**Prerequisiti**
- Python 3.11.9
- Account Hugging Face (per accedere al modello)
- Accesso a una GPU A100 da circa 50GB (consigliato)
- PDF da analizzare (`allProva2.pdf` nella root del progetto)

  
 **Nota per l'esecuzione**
 
- I passaggi sono pensati per essere eseguiti da terminale (cmd o shell). Ricorda di modificare opportunamente i percorsi delle directory se necessario.
- Il funzionamento descritto è stato testato con successo nell’ambiente `llama3-env` su Jupyter GPU fornito dall’università.


**Setup Ambiente Virtuale**

Puoi usare un nome qualsiasi per l’ambiente virtuale (ad esempio `llama_env`, `venv`, `mio_env`):
```bash
python -m venv llama_env
source llama_env/bin/activate  # Su Windows: llama_env\Scripts\activate
```

**Installazione Dipendenze** 

Assicurati di essere nell’ambiente virtuale, quindi installa i pacchetti necessari:
```bash
pip install -r requirements.txt
```

**Autenticazione Hugging Face**

Accedi al tuo account Hugging Face per poter scaricare il modello:
```bash
huggingface-cli login
```

**Scarica il Modello**

Questo comando scaricherà il modello Mistral-7B-Instruct-v0.2 e lo salverà nella cache locale (`~/.cache/huggingface`):
```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM;
tokenizer = AutoTokenizer.from_pretrained('Mistral-7B-Instruct-v0.2', use_auth_token=True);
model = AutoModelForCausalLM.from_pretrained('Mistral-7B-Instruct-v0.2', use_auth_token=True)"
```

**Estrazione del Testo dal PDF**

Posiziona il file `allProva2.pdf` nella directory principale del progetto.
Estrai il testo, dividilo in chunk e salva il risultato in `./data/BigData`:
```bash
python estrazione.py
```

**Costruzione dell’Indice Semantico (Chroma)**

Esegui il seguente script per creare l’indice semantico basato sui chunk di testo:
```bash
python chroma.py
```

**Avvio del Server FastAPI**
Assicurati che lo script `server.py` usi `device="cuda"` se vuoi sfruttare la GPU:
```bash
python server.py
```
Il server FastAPI sarà avviato e pronto a ricevere richieste.

**Interrogare il Modello**

E' stato usato il file `Test.ipynb` per porre domande in linguaggio naturale e ottenere risposte basate sul contenuto del PDF.


