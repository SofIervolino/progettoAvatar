import os
import logging
import re
from typing import List

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.api import EmbeddingFunction

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

#################################
# LOGGING
#################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#################################
# 1) Inizializzazione FastAPI + UI
#################################
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
PORT = int(os.environ.get("PORT", 8000))

#################################
# 2) Caricamento Modello LLM (senza fine-tuning)
#################################
model_path = "mistralai/Mistral-7B-Instruct-v0.2" #Cambia col modello open-source che preferisci
hf_token = "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" #Token Hugging Face se serve

logger.info(f"Carico il modello pre-addestrato: {model_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=hf_token)
    
    # RISOLVE IL PROBLEMA DEL PADDING
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", use_auth_token=hf_token).to("cuda:5") 
    model.resize_token_embeddings(len(tokenizer))  # Aggiorna la dimensione del vocabolario

except Exception as e:
    logger.error(f"Errore nel caricamento del modello: {e}")
    raise e

#################################
# 3) Funzione di generazione
#################################
def strip_tags(text: str) -> str:
    """
    Elimina eventuali tag come 'Domanda:', 'Contesto:', 'Risposta:' e riduce spazi multipli.
    """
    text = re.sub(r"(Domanda:|Contesto:|Risposta:)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_answer(question: str, context: str) -> str:
    """
    Prompt conciso + parametri 'stretti' per risposte brevi.
    """
    prompt = (
        "Sei un professore di Big Data. Rispondi in modo breve e preciso, massimo 5 frasi. "
        "NON inventare nulla, NON menzionare immagini o tabelle. Completa sempre l'utlima frase e dai una risposta di senso compiuto. Se la risposta non si trova "
        "nel contesto, di' chiaramente che mancano informazioni.\n\n"
        f"Domanda: {question}\n"
        f"Contesto:\n{context}\n\n"
        "Risposta:"
    )

    logger.info(f"DEBUG - Prompt inviato al modello:\n{prompt[:500]}...\n")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    # Parametri "stretti" per risposte brevi e meno tempo
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=120,   # ridurre se vuoi ancora meno testo
        num_beams=2,
        early_stopping=True,
        temperature=0.2,
        no_repeat_ngram_size=3,
        top_k=20,
        top_p=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Estrazione parte "Risposta:"
    if "Risposta:" in gen_text:
        splitted = gen_text.split("Risposta:", 1)[-1].strip()
    else:
        splitted = gen_text.strip()
    final_text = finalize_sentence(splitted)
    return final_text

#################################
# 4) ChromaDB: retrieval
#################################
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="multi-qa-MiniLM-L6-cos-v1", device="cuda:5"): 
        self.model = SentenceTransformer(model_name, device=device)
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

PERSIST_DIR = "/home/jovyan/shared/tesista1/chroma_v3"

def get_chroma_client() -> chromadb.Client:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    if not hasattr(get_chroma_client, "client"):
        get_chroma_client.client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(persist_directory=PERSIST_DIR),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        logger.info(f"Chroma PersistentClient inizializzato con persist_directory: {PERSIST_DIR}")
    return get_chroma_client.client
    
def get_collection():
    client = get_chroma_client()
    embedding_fn = MyEmbeddingFunction()
    coll = client.get_or_create_collection(
        name="bigdata_collection",
        embedding_function=embedding_fn
    )
    return coll
    
def retrieve_chunks(question: str, top_k=5, min_score=0.5):
    """
    Recupera chunk più pertinenti e filtra con score >= min_score.
    Limita a 3 chunk max per ridurre prompt e tempi.
    """
    coll = get_collection()
    results = coll.query(query_texts=[question], n_results=top_k)
    
    docs = results.get("documents", [[]])[0]
    scores = results.get("distances", [[]])[0]
    
    chunk_score_pairs = []
    for doc, dist in zip(docs, scores):
        relevance = 1.0 / (1.0 + dist)  # distanza -> pseudo-score
        if relevance >= min_score:
            chunk_score_pairs.append((doc, relevance))
    
    # Ordina e prendi i primi 3
    chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
    chunk_score_pairs = chunk_score_pairs[:3]  # max 3 chunk

    return chunk_score_pairs

def finalize_sentence(text: str) -> str:
    """
    Rimuove l'eventuale parte di frase incompleta alla fine, 
    tronca all'ultimo segno di punteggiatura. 
    Se non trova alcun segno, taglia l'ultima parola e aggiunge un punto finale.
    """
    text = text.strip()
    # Cerca l'ultimo punto, esclamativo o interrogativo
    last_dot = text.rfind(".")
    last_excl = text.rfind("!")
    last_quest = text.rfind("?")

    last_punc = max(last_dot, last_excl, last_quest)

    if last_punc == -1:
        # Nessuna punteggiatura trovata: 
        # 1) tagliamo l'ultima parola e 
        # 2) aggiungiamo un punto
        words = text.split()
        if not words:
            return text  # testo vuoto
        # Rimuovi l'ultima parola monca
        trimmed = " ".join(words[:-1])
        if not trimmed:
            return text + "."  # se c'era solo una parola, appendo .
        return trimmed + "."
    else:
        # Tronca tutto ciò che segue la punteggiatura
        return text[:last_punc+1].strip()


#################################
# 5) Endpoints
#################################
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "PORT": PORT})

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    """
    Endpoint per domande: unisce max 3 chunk, chiama generate_answer e restituisce.
    """
    user_q = req.question.strip()
    if not user_q:
        raise HTTPException(status_code=400, detail="Domanda vuota.")

    scored_chunks = retrieve_chunks(user_q, top_k=5)

    if not scored_chunks:
        return {
            "answer": "Le informazioni fornite non sono sufficienti per rispondere alla domanda.",
            "chunks_used": []
        }

    # Unisce i chunk senza [Rilevanza=..]
    context_aggregato = " ".join(chunk for chunk, _ in scored_chunks)
    logger.info(f"[RAG] Contesto unito (max 3 chunk): {context_aggregato[:100]}...")

    answer = generate_answer(user_q, context_aggregato)

    return {
        "answer": answer,
        "chunks_used": [{"chunk": c, "score": s} for c, s in scored_chunks]
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=True)
