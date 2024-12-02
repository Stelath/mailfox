import os
import imaplib
import datetime
import email
import hashlib
from email.header import decode_header
import pandas as pd
import re
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

class EmailHandler():
    def __init__(self, username, password):
        self.mail = imaplib.IMAP4_SSL("imap.gmail.com")
        self.mail.login(username, password)
        self.mail.select("INBOX")

    def get_all_mail_uids(self):
        result, data = self.mail.uid('search', None, "ALL")
        return [d.decode('utf-8') for d in data[0].split()]

    def format_folders(self, folders, plain=False):
        formatted_folders = []
        for folder in folders:
            if not plain:
                folder = f'"{folder}"' if folder[0] != '"' else folder
            else:
                folder = folder[1:-1] if folder[0] == '"' else folder
            formatted_folders.append(folder)
        return formatted_folders

    def get_all_folders(self):
        result, data = self.mail.list()
        return [d.decode('utf-8').split(' "/" ')[1] for d in data]

    def get_subfolders(self, folders):
        folders = self.format_folders(folders, plain=True)
        all_folders = self.get_all_folders()
        subfolders = [folder for folder in all_folders if any(f in folder for f in folders)]
        return subfolders

    def get_folder_uids(self, folder):
        result, data = self.mail.select(folder)
        if result == 'OK':
            result, data = self.mail.uid('search', None, "ALL")
            return [d.decode('utf-8') for d in data[0].split()]
        else:
            print(f"Failed to select folder {folder}")
            return []

    def hash_email(self, email):
        hash_string = email['from'] + email['to'] + email['subject'] + email['date']
        uuid = hashlib.sha256(hash_string.encode()).hexdigest()
        return uuid

    def get_mail(self, filter='unseen', *, uids=None, folders=["INBOX"], return_dataframe=True):
        folders = self.format_folders(folders)
        emails = []

        if not folders:
            print("No folders to read.")
            return

        uids = {}
        for folder in folders:
            result, data = self.mail.select(folder)
            if result == 'OK':
                if filter == 'unseen':
                    result, data = self.mail.uid('search', None, "(UNSEEN)")
                elif filter == 'seen':
                    result, data = self.mail.uid('search', None, "(SEEN)")
                elif filter == 'all':
                    result, data = self.mail.uid('search', None, "ALL")
                elif filter == 'uids' and uids is not None:
                    result = 'OK'
                    data = [uid.encode('utf-8') for uid in uids]
                else:
                    print("Invalid filter. Please use 'unseen', 'all', or 'uids'.")
                    return
                uids[folder] = data[0].split()
            else:
                print(f"Failed to select folder {folder}")
                continue

        if not uids:
            print("No new emails to read.")
            return pd.DataFrame([]) if return_dataframe else []

        for folder, folder_uids in tqdm(uids.items(), desc="Processing Folders", position=0, leave=False):
            self.mail.select(folder)
            for num in tqdm(folder_uids, desc=f"Getting Emails from {folder}", position=1, leave=False):
                result, email_data = self.mail.uid('fetch', num, '(BODY.PEEK[])')
                try:
                    raw_email = email_data[0][1]
                    raw_email_string = raw_email.decode('utf-8')
                except:
                    continue
                email_message = email.message_from_string(raw_email_string)

                date_tuple = email.utils.parsedate_tz(email_message['Date'])
                if date_tuple:
                    local_date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
                    local_message_date = local_date.strftime("%a, %d %b %Y %H:%M:%S")

                try:
                    email_from = str(decode_header(email_message['From'])[0][0])
                    email_to = str(decode_header(email_message['To'])[0][0])
                    subject = str(decode_header(email_message['Subject'])[0][0])
                    uuid = self.hash_email({'from': email_from, 'to': email_to, 'subject': subject, 'date': local_message_date})
                    uid = num.decode('utf-8')
                    message_id = email_message['Message-ID']
                except:
                    continue

                body = ""
                if email_message.is_multipart():
                    for part in email_message.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        if content_type == 'text/plain' and 'attachment' not in content_disposition:
                            try:
                                body += part.get_payload(decode=True).decode("utf-8", errors="ignore")
                            except:
                                body += part.get_payload(decode=True).decode("latin-1", errors="ignore")
                else:
                    try:
                        body = email_message.get_payload(decode=True).decode("utf-8", errors="ignore")
                    except:
                        body = email_message.get_payload(decode=True).decode("latin-1", errors="ignore")

                raw_body = body
                processed_body = self._process_body(body)
                paragraphs = self._split_into_paragraphs(processed_body)

                if not paragraphs:
                    continue

                emails.append({
                    'uid': uid,
                    'folder': folder,
                    'uuid': uuid,
                    'from': email_from,
                    'to': email_to,
                    'subject': subject,
                    'date': local_message_date,
                    'message_id': message_id,
                    'paragraphs': paragraphs,
                    'raw_body': raw_body
                })

        return pd.DataFrame(emails) if return_dataframe else emails

    def _process_body(self, body):
        # Remove HTML tags and non-textual elements
        soup = BeautifulSoup(body, 'html.parser')
        for element in soup(["script", "style", "img", "table", "code"]):
            element.decompose()
        text = soup.get_text()

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Handle special characters
        text = text.encode('ascii', 'ignore').decode('ascii')

        return text.strip()

    def _split_into_paragraphs(self, text):
        paragraphs = [para.strip() for para in text.split('\n') if para.strip()]
        return paragraphs

    def create_folder(self, folder):
        result = self.mail.create(folder)
        if result[0] == 'OK':
            print(f"Created folder {folder}")
        else:
            print(f"Failed to create folder {folder}")

    def move_mail(self, uids, folder):
        folder = self.format_folders([folder])[0]
        for uid in uids:
            result = self.mail.uid('COPY', uid, folder)
            if result[0] == 'OK':
                self.mail.uid('STORE', uid, '+FLAGS', '(\Deleted)')
                self.mail.expunge()
            else:
                print(f"Failed to move {uid} to {folder}")

    def delete_mail(self, uids):
        self.move_mail(uids, "INBOX.Trash")

    def get_message_id_folder_mapping(self, folders):
        message_id_to_folder = {}
        for folder in tqdm(folders, desc="Fetching Message-IDs"):
            result, data = self.mail.select(folder)
            if result != 'OK':
                continue
            result, data = self.mail.uid('search', None, "ALL")
            if result != 'OK':
                continue
            uids = data[0].split()
            if not uids:
                continue
            for uid in uids:
                result, email_data = self.mail.uid('fetch', uid, '(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)])')
                if result != 'OK':
                    continue
                raw_email = email_data[0][1]
                email_message = email.message_from_bytes(raw_email)
                message_id = email_message['Message-ID']
                if message_id:
                    message_id_to_folder[message_id] = folder
        return message_id_to_folder