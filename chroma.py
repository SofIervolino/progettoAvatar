import os
import nltk
import logging
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.api import EmbeddingFunction
from sentence_transformers import SentenceTransformer

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Scarica i pacchetti NLTK necessari
nltk.download('punkt', quiet=True)

##################################
# 1) Funzione di embedding personalizzata
##################################
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1", device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

##################################
# 2) Funzione per costruire l'indice Chroma persistente
##################################
def build_chroma_index(txt_folder: str, collection_name: str, persist_dir: str):
    # Assicura la creazione della directory di persistenza
    os.makedirs(persist_dir, exist_ok=True)
    
    # Crea le Settings specificando la directory di persistenza
    settings = Settings(persist_directory=persist_dir)
    
    # Inizializza il PersistentClient (i dati verranno salvati su disco)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=settings,
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    
    embedding_fn = MyEmbeddingFunction()
    
    # Crea (o recupera) la collezione con il nome specificato
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    
    # Cerca i file .txt nella cartella indicata
    txt_folder = Path(txt_folder)
    txt_files = list(txt_folder.glob("*.txt"))
    if not txt_files:
        logger.warning(f"Nessun file .txt trovato in {txt_folder}")
        return

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Errore nella lettura di {txt_file}: {e}")
            continue
        
        # Aggiunge il documento (l'intero contenuto del file)
        collection.add(
            documents=[text],
            ids=[txt_file.stem],
            metadatas=[{"source": str(txt_file)}]
        )
    
    logger.info(f"Collezione '{collection_name}' creata/aggiornata in '{persist_dir}'.")
    
    # Per il debug, stampiamo le collezioni e il conteggio dei documenti
    colls_info = client.list_collections()
    print("Elenco collezioni:", colls_info)
    for coll in colls_info:
        coll_obj = client.get_collection(name=coll.name, embedding_function=embedding_fn)
        print(f" -> {coll_obj.name}: {coll_obj.count()} documenti")
    
    # Il client persistente salva automaticamente i dati al termine dell'esecuzione

if __name__ == "__main__":
    build_chroma_index(
        txt_folder="./data/BigData",
        collection_name="bigdata_collection",
        persist_dir="/home/jovyan/shared/tesista1/chroma_v3"
    )
