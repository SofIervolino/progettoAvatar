import os
import re
import pdfplumber
##import nltk
##nltk.data.path.append("/home/jovyan/shared/tesista1/nltk_data")
import nltk
nltk.download('punkt', quiet=True)

#############################################
# 1) Regex e filtri su righe
#############################################
CHAPTER_REGEX = re.compile(r"(?i)^\s*(capitolo|chapter)\s+\d+")
FIGURE_REGEX  = re.compile(r"(?i)^\s*(figura|figure|fig\.?|immagine)\s*\d+.*")

def remove_chapter_indices(line: str) -> bool:
    """True se la linea è qualcosa tipo 'Capitolo 1'."""
    return bool(CHAPTER_REGEX.search(line))

def remove_figure_references(line: str) -> bool:
    """True se la linea è un riferimento a figura/immagine."""
    return bool(FIGURE_REGEX.search(line))

def clean_line(line: str) -> str:
    """
    - Rimuove ligature comuni (es. 'ﬁ' -> 'fi')
    - Riduce spazi multipli
    - Trim spazi a inizio/fine
    """
    line = line.replace("ﬁ", "fi")  # Esempio di sostituzione ligature
    # Altre sostituzioni possibili: "ﬀ" -> "ff", "ﬂ" -> "fl", ecc.

    # Riduzione spazi multipli
    line = re.sub(r"\s+", " ", line).strip()
    return line

#############################################
# 2) Estrazione e pulizia del testo con pdfplumber
#############################################
def extract_and_clean_text(pdf_path: str) -> str:
    """
    - Estrae testo lineare da ogni pagina
    - Rimuove righe di capitolo/figure
    - Pulizia di spazi e ligature
    - Restituisce un'unica stringa
    """
    all_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if not page_text:
                continue

            lines = page_text.split('\n')
            for line in lines:
                line = clean_line(line)
                if not line:
                    continue
                # Salta linee come "Capitolo X" o "Figura X..."
                if remove_chapter_indices(line):
                    continue
                if remove_figure_references(line):
                    continue

                all_lines.append(line)

    # Concatena con newline
    text = " ".join(all_lines)
    return text

#############################################
# 3) Chunking semantico (paragrafi e frasi)
#############################################
def chunk_text_semantic(text: str, max_tokens: int = 100) -> list:
    """
    Esempio di suddivisione "semantica":
     1) Divide in paragrafi (usando newline doppi)
     2) Ogni paragrafo in frasi con nltk.sent_tokenize
     3) Accumula le frasi in chunk finché non si supera max_tokens
    """
    # Divisione in paragrafi: se il PDF conserva almeno qualche riga vuota
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Tokenizza in frasi
        sentences = nltk.sent_tokenize(para)
        
        for sent in sentences:
            tokens_in_sent = nltk.word_tokenize(sent)
            n_tokens = len(tokens_in_sent)

            # Se l'aggiunta di questa frase sforerebbe max_tokens, chiudi chunk e riparti
            if current_count + n_tokens > max_tokens:
                if current_chunk:
                    chunk_str = " ".join(current_chunk)
                    chunks.append(chunk_str)
                
                # Avvia un nuovo chunk con la frase corrente
                current_chunk = [sent]
                current_count = n_tokens
            else:
                # Continua ad accumulare
                current_chunk.append(sent)
                current_count += n_tokens

    # Ultimo chunk
    if current_chunk:
        chunk_str = " ".join(current_chunk)
        chunks.append(chunk_str)

    return chunks

#############################################
# 4) Salvataggio su file multipli
#############################################
def save_chunks_to_folder(chunks: list, out_folder: str):
    """
    Salva ciascun chunk in un file .txt separato nella cartella out_folder.
    Esempio di naming: chunk_0.txt, chunk_1.txt, ...
    """
    os.makedirs(out_folder, exist_ok=True)

    for i, chunk in enumerate(chunks):
        filename = f"chunk_{i}.txt"
        filepath = os.path.join(out_folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chunk)
        print(f"Salvato: {filepath}  ({len(chunk.split())} parole)")

from sentence_transformers import SentenceTransformer

class E5Embedder:
    def __init__(self, model_name="intfloat/multilingual-e5-base", device="cuda"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode_chunks(self, chunks: list) -> list:
        # Il modello E5 richiede prefisso "passage: " per l'embedding dei testi
        passages = [f"passage: {chunk}" for chunk in chunks]
        embeddings = self.model.encode(passages, convert_to_tensor=False)
        return embeddings


#############################################
# ESEMPIO MAIN
#############################################
if __name__ == "__main__":
    pdf_file = "allProva2.pdf"       # Nome PDF da processare
    out_dir = "./data/BigData"       # Cartella output
    max_tokens = 100                 # Soglia massima per chunk

    # 1) Estrae e pulisce il testo
    text_pulito = extract_and_clean_text(pdf_file)

    # 2) Chunk su base "semantica"
    chunks = chunk_text_semantic(text_pulito, max_tokens=max_tokens)
    print(f"Trovati {len(chunks)} chunk totali.")

    # 3) Salva i chunk su file
    save_chunks_to_folder(chunks, out_dir)

    # 4) Genera gli embedding con E5
    embedder = E5Embedder()
    embeddings = embedder.encode_chunks(chunks)
    print(f"Generati {len(embeddings)} embedding con dimensione {len(embeddings[0])}")



