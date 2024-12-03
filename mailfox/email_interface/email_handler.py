import os
import datetime
import email
import hashlib
from email.header import decode_header
import pandas as pd
import re
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import time
from imapclient import IMAPClient
from threading import Thread, Event


class EmailHandler:
    def __init__(self, username, password, server="imap.gmail.com", ssl=True):
        self.server = server
        self.username = username
        self.password = password
        self.ssl = ssl
        self.mail = IMAPClient(self.server, use_uid=True, ssl=self.ssl)
        self.mail.login(self.username, self.password)
        self.idle_event = Event()

    def format_folders(self, folders):
        # No special formatting needed with IMAPClient
        return folders

    def get_all_folders(self):
        folders = self.mail.list_folders()
        return [folder[-1] for folder in folders]

    def get_subfolders(self, folders):
        all_folders = self.get_all_folders()
        subfolders = [folder for folder in all_folders if any(f in folder for f in folders)]
        return subfolders

    def idle_check_new_mail(self, folders, callback):
        def idle_worker(folder):
            while not self.idle_event.is_set():
                try:
                    self.mail.select_folder(folder)
                    self.mail.idle()
                    print(f"Started IDLE mode for folder: {folder}")
                    responses = self.mail.idle_check(timeout=300)
                    for response in responses:
                        if response[1] == b'EXISTS':
                            print(f"New email in folder: {folder}")
                            callback(folder)
                except Exception as e:
                    print(f"Error in IDLE for folder {folder}: {e}")
                    time.sleep(5)
                finally:
                    self.mail.idle_done()

        threads = []
        for folder in folders:
            thread = Thread(target=idle_worker, args=(folder,))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        try:
            while not self.idle_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.idle_event.set()
            for thread in threads:
                thread.join()

    def get_mail(self, filter='unseen', folders=["INBOX"], return_dataframe=True):
        emails = []
        for folder in tqdm(folders, desc="Processing Folders", position=0, leave=False):
            self.mail.select_folder(folder)
            if filter == 'unseen':
                messages = self.mail.search(['UNSEEN'])
            elif filter == 'seen':
                messages = self.mail.search(['SEEN']) 
            elif filter == 'all':
                messages = self.mail.search('ALL')
            else:
                print("Invalid filter. Please use 'unseen', 'all', or 'seen'.")
                return

            for uid in tqdm(messages, desc=f"Getting Emails from {folder}", position=1, leave=False):
                # Use correct PEEK format for IMAPClient
                raw_email = self.mail.fetch(uid, ['BODY.PEEK[]', 'FLAGS'])
                email_message = email.message_from_bytes(raw_email[uid][b'BODY[]'])
                flags = raw_email[uid][b'FLAGS']
                mail = self._process_email(uid, folder, email_message, flags)
                if mail:
                    emails.append(mail)

        return pd.DataFrame(emails) if return_dataframe else emails

    def _process_email(self, uid, folder, email_message, flags):
        try:
            date_tuple = email.utils.parsedate_tz(email_message['Date'])
            if date_tuple:
                local_date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
                local_message_date = local_date.strftime("%a, %d %b %Y %H:%M:%S")

            email_from = str(decode_header(email_message['From'])[0][0])
            email_to = str(decode_header(email_message['To'])[0][0])
            subject = str(decode_header(email_message['Subject'])[0][0])
            uuid = self.hash_email({'from': email_from, 'to': email_to, 'subject': subject, 'date': local_message_date})
            message_id = email_message['Message-ID']
        except Exception as e:
            print(f"Error processing email: {e}")
            return None

        body = self._extract_body(email_message)
        if not body:
            return None

        raw_body = body
        processed_body = self._process_body(body)
        paragraphs = self._split_into_paragraphs(processed_body)

        if not paragraphs:
            return None

        return {
            'uid': uid,
            'folder': folder,
            'uuid': uuid,
            'from': email_from,
            'to': email_to,
            'subject': subject,
            'date': local_message_date,
            'message_id': message_id,
            'paragraphs': paragraphs,
            'raw_body': raw_body,
            'flags': flags
        }

    def _extract_body(self, email_message):
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
        return body

    def hash_email(self, email_dict):
        hash_string = email_dict['from'] + email_dict['to'] + email_dict['subject'] + email_dict['date']
        uuid = hashlib.sha256(hash_string.encode()).hexdigest()
        return uuid

    def _process_body(self, body):
        soup = BeautifulSoup(body, 'html.parser')
        for element in soup(["script", "style", "img", "table", "code"]):
            element.decompose()
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()

    def _split_into_paragraphs(self, text):
        paragraphs = [para.strip() for para in text.split('\n') if para.strip()]
        return paragraphs

    def move_mail(self, uids, folder):
        folder = folder.replace('"', '').strip()
        self.mail.move(uids, folder)

    def delete_mail(self, uids):
        self.mail.delete_messages(uids)
        self.mail.expunge()