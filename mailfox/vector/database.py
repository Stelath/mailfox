import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from tqdm.auto import tqdm
from enum import Enum
import sqlite3
import textwrap
import os
import tiktoken
from ..core.auth import read_credentials

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
        self.embedding_function_type = embedding_function
        if embedding_function == EmbeddingFunctions.OPENAI:
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
                
            self.default_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")
            self.max_tokens = MAX_TOKENS["text-embedding-3-small"]
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
        else:
            self.default_ef = embedding_functions.DefaultEmbeddingFunction()
            self.max_tokens = MAX_TOKENS["all-MiniLM-L6-v2"]
            self.tokenizer = None
        
        self.emails_collection = self.chroma_client.get_or_create_collection(name="emails", embedding_function=self.default_ef)
        self.email_db_path = os.path.join(db_path, "emails.db")
        self._init_email_db()

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if self.embedding_function_type == EmbeddingFunctions.OPENAI:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate for sentence transformers
            return len(text.split())

    def _chunk_text(self, text: str) -> list[str]:
        """Break text into chunks that fit within token limit."""
        if not text:
            return []

        if self.embedding_function_type == EmbeddingFunctions.OPENAI:
            # Split into sentences first
            sentences = text.replace('\n', ' ').split('. ')
            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence = sentence.strip() + '.'
                sentence_length = self._count_tokens(sentence)

                # If single sentence is too long, split it
                if sentence_length > self.max_tokens:
                    words = sentence.split()
                    temp_chunk = []
                    temp_length = 0
                    for word in words:
                        word_length = self._count_tokens(word + ' ')
                        if temp_length + word_length > self.max_tokens:
                            chunks.append(' '.join(temp_chunk))
                            temp_chunk = [word]
                            temp_length = word_length
                        else:
                            temp_chunk.append(word)
                            temp_length += word_length
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                    continue

                # Try to add sentence to current chunk
                if current_length + sentence_length <= self.max_tokens:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length

            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks
        else:
            # For sentence transformers, use simpler chunking
            return textwrap.wrap(text, width=self.max_tokens * 4, break_long_words=True)

    def embed_paragraphs(self, paragraphs: list[str]):
        chunked_paragraphs = []
        for p in paragraphs:
            chunks = self._chunk_text(p)
            chunked_paragraphs.extend(chunks)
        
        if not chunked_paragraphs:
            return []
            
        # For OpenAI, ensure no chunk exceeds the token limit
        if self.embedding_function_type == EmbeddingFunctions.OPENAI:
            chunked_paragraphs = [chunk for chunk in chunked_paragraphs 
                                if self._count_tokens(chunk) <= self.max_tokens]
            
        if not chunked_paragraphs:
            return []
            
        embeddings = self.embed(chunked_paragraphs)
        return embeddings

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
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS seen_uids (
                uid INTEGER,
                folder TEXT,
                PRIMARY KEY (uid, folder)
            )
        ''')
        self.conn.commit()

    def is_emails_empty(self):
        docs = self.emails_collection.get(include=[])
        return len(docs['ids']) == 0

    def embed(self, text: list[str]):
        return self.default_ef(text)

    def store_emails(self, emails: list[dict]):
        for idx, mail in enumerate(tqdm(emails, desc="Saving Emails to Database")):
            try:
                uuid = mail['uuid']
                paragraphs = mail.get('paragraphs', [])
                if not paragraphs:
                    continue

                # Check if email exists in ChromaDB first
                test_ids = [f"{uuid}_0"]  # Check first paragraph ID
                existing = self.emails_collection.get(
                    ids=test_ids,
                    include=[]
                )
                email_exists = len(existing['ids']) > 0

                # Store email metadata in SQLite database
                self.cursor.execute('''
                    INSERT OR REPLACE INTO emails (uuid, uid, folder, sender, recipient, subject, date, message_id, raw_body)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (uuid, mail['uid'], mail['folder'], mail['from'], mail['to'], mail['subject'], mail['date'], mail['message_id'], mail['raw_body']))
                self.conn.commit()

                if email_exists:
                    # Only update metadata if email exists
                    ids = [f"{uuid}_{i}" for i in range(len(paragraphs))]
                    metadatas = [{'uuid': uuid, 'folder': mail['folder'], 'paragraph_index': i} for i in range(len(paragraphs))]
                    existing_ids = set(self.emails_collection.get(
                        ids=ids,
                        include=[]
                    )['ids'])
                    for i, vec_id in enumerate(ids):
                        if vec_id in existing_ids:
                            self.emails_collection.update(
                                ids=[vec_id],
                                metadatas=[metadatas[i]]
                            )
                else:
                    # Only calculate embeddings for new emails
                    try:
                        embeddings = self.embed_paragraphs(paragraphs)
                        if embeddings:
                            ids = [f"{uuid}_{i}" for i in range(len(embeddings))]
                            metadatas = [{'uuid': uuid, 'folder': mail['folder'], 'paragraph_index': i} for i in range(len(embeddings))]
                            self.emails_collection.add(
                                ids=ids,
                                embeddings=embeddings,
                                metadatas=metadatas
                            )
                    except Exception as e:
                        print(f"Error embedding new email {uuid}: {e}")
            except Exception as e:
                print(f"Error storing embeddings for email {uuid}: {e}")

    def get_all_emails(self):
        """Retrieve all emails from the SQLite database."""
        self.cursor.execute('''
            SELECT uuid, uid, folder, sender, recipient, subject, date, message_id, raw_body 
            FROM emails
        ''')
        rows = self.cursor.fetchall()
        
        emails = []
        for row in rows:
            email = {
                'uuid': row[0],
                'uid': row[1],
                'folder': row[2],
                'from': row[3],
                'to': row[4],
                'subject': row[5],
                'date': row[6],
                'message_id': row[7],
                'raw_body': row[8],
                'paragraphs': self._get_paragraphs_from_raw_body(row[8])
            }
            emails.append(email)
        
        return emails

    def _get_paragraphs_from_raw_body(self, raw_body):
        """Extract paragraphs from raw body text."""
        if not raw_body:
            return []
        
        # Split by double newlines to get paragraphs
        paragraphs = [p.strip() for p in raw_body.split('\n\n')]
        # Filter out empty paragraphs
        paragraphs = [p for p in paragraphs if p]
        return paragraphs

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

    def add_seen_uids(self, uids, folder):
        """Store UIDs that have been seen in a folder."""
        try:
            self.cursor.executemany(
                'INSERT OR IGNORE INTO seen_uids (uid, folder) VALUES (?, ?)',
                [(int(uid), folder) for uid in uids]
            )
            self.conn.commit()
        except Exception as e:
            print(f"Error storing seen UIDs: {e}")

    def check_seen_uids(self, uids, folder):
        """Check which UIDs have been seen before in a folder."""
        try:
            seen_uids = set()
            for uid in uids:
                self.cursor.execute(
                    'SELECT uid FROM seen_uids WHERE uid = ? AND folder = ?',
                    (int(uid), folder)
                )
                if self.cursor.fetchone():
                    seen_uids.add(int(uid))
            return seen_uids
        except Exception as e:
            print(f"Error checking seen UIDs: {e}")
            return set()

    def get_seen_uids(self):
        """Get all seen UIDs and their folders."""
        try:
            self.cursor.execute('SELECT uid, folder FROM seen_uids')
            rows = self.cursor.fetchall()
            folder_uids = {}
            for uid, folder in rows:
                if folder not in folder_uids:
                    folder_uids[folder] = set()
                folder_uids[folder].add(int(uid))
            return folder_uids
        except Exception as e:
            print(f"Error getting seen UIDs: {e}")
            return {}

    def close(self):
        self.conn.close()
