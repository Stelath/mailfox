import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from tqdm.auto import tqdm
from enum import Enum
import sqlite3
import textwrap
import os


class EmbeddingFunctions(str, Enum):
    SENTENCE_TRANSFORMER = "st"
    OPENAI = "openai"

MAX_TOKENS = {
    "text-embedding-3-small": 8191,
    "all-MiniLM-L6-v2": 384
}

class VectorDatabase():
    def __init__(self, db_path="./data/", *, embedding_function=None, openai_api_key=None):
        self.chroma_client = chromadb.PersistentClient(os.path.join(db_path, "chroma"))
        if embedding_function == EmbeddingFunctions.OPENAI:
            self.default_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")
            self.max_tokens = MAX_TOKENS["text-embedding-3-small"]
        else:
            self.default_ef = embedding_functions.DefaultEmbeddingFunction()
            self.max_tokens = MAX_TOKENS["all-MiniLM-L6-v2"]
        
        self.emails_collection = self.chroma_client.get_or_create_collection(name="emails", embedding_function=self.default_ef)
        self.email_db_path = os.path.join(db_path, "emails.db")
        self._init_email_db()

    def _init_email_db(self):
        self.conn = sqlite3.connect(self.email_db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS emails (
                uuid TEXT PRIMARY KEY,
                uid TEXT,
                folder TEXT,
                sender TEXT,
                recipient TEXT,
                subject TEXT,
                date TEXT,
                message_id TEXT,
                raw_body TEXT
            )
        ''')
        self.conn.commit()

    def is_emails_empty(self):
        docs = self.emails_collection.get(include=[])
        return len(docs['ids']) == 0

    def embed(self, text: list[str]):
        return self.default_ef(text)

    def _chunk_text(self, text: str) -> list[str]:
        """Break text into chunks that fit within token limit"""
        # Estimate ~4 chars per token
        chars_per_chunk = self.max_tokens * 4
        chunks = textwrap.wrap(text, width=chars_per_chunk, break_long_words=True)
        return chunks

    def embed_paragraphs(self, paragraphs: list[str]):
        chunked_paragraphs = []
        for p in paragraphs:
            chunks = self._chunk_text(p)
            chunked_paragraphs.extend(chunks)
            
        embeddings = self.embed(chunked_paragraphs)
        return embeddings

    def store_emails(self, emails: list[dict]):
        for idx, mail in enumerate(tqdm(emails, desc="Saving Emails to Database")):
            try:
                paragraphs = mail.get('paragraphs', [])
                if not paragraphs:
                    continue
                
                # Get embeddings for chunked paragraphs
                embeddings = self.embed_paragraphs(paragraphs)
            except Exception as e:
                print(f"Error embedding email {idx}: {e}")
                continue

            uuid = mail['uuid']

            # Store email metadata in SQLite database
            self.cursor.execute('''
                INSERT OR REPLACE INTO emails (uuid, uid, folder, sender, recipient, subject, date, message_id, raw_body)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (uuid, mail['uid'], mail['folder'], mail['from'], mail['to'], mail['subject'], mail['date'], mail['message_id'], mail['raw_body']))
            self.conn.commit()

            # Store embeddings in ChromaDB
            ids = [f"{uuid}_{i}" for i in range(len(embeddings))]
            metadatas = [{'uuid': uuid, 'folder': mail['folder'], 'paragraph_index': i} for i in range(len(embeddings))]
            self.emails_collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def get_all_embeddings(self):
        docs = self.emails_collection.get(include=['embeddings'])
        ids = docs['ids']
        embeddings = np.array(docs['embeddings'])
        return {'ids': ids, 'embeddings': embeddings}

    def get_email_by_uuid(self, uuid):
        self.cursor.execute('SELECT * FROM emails WHERE uuid=?', (uuid,))
        return self.cursor.fetchone()

    def update_email_folder(self, uuid, new_folder):
        self.cursor.execute('UPDATE emails SET folder=? WHERE uuid=?', (new_folder, uuid))
        self.conn.commit()

    def close(self):
        self.conn.close()