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
        self.stop_event = Event()
        self._uid_validity_cache = {}

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

    def _check_uid_validity(self, folder):
        """Check if folder's UID validity has changed."""
        try:
            self.mail.select_folder(folder)
            current_validity = self.mail.folder_status(folder)[b'UIDVALIDITY']
            cached_validity = self._uid_validity_cache.get(folder)
            
            if cached_validity is None:
                self._uid_validity_cache[folder] = current_validity
                return True
            
            if current_validity != cached_validity:
                print(f"UID validity changed for folder {folder}")
                self._uid_validity_cache[folder] = current_validity
                return False
                
            return True
        except Exception as e:
            print(f"Error checking UID validity for {folder}: {e}")
            return True

    def _recache_folder(self, folder, limit=None):
        """Recache emails in a folder after UID validity change."""
        try:
            self.mail.select_folder(folder)
            if limit:
                # Get most recent emails up to limit
                messages = self.mail.search(['ALL'])[-limit:]
            else:
                # Get all emails
                messages = self.mail.search(['ALL'])
            
            return self.get_mail(filter='all', folders=[folder], uids=messages, return_dataframe=True)
        except Exception as e:
            print(f"Error recaching folder {folder}: {e}")
            return pd.DataFrame()

    def poll_folders(self, folders, folder_uids, callback, enable_uid_validity=True, recache_limit=100):
        """Poll folders for changes."""
        for folder in folders:
            try:
                # Check UID validity if enabled
                if enable_uid_validity and not self._check_uid_validity(folder):
                    print(f"Recaching folder {folder}")
                    emails = self._recache_folder(folder, recache_limit)
                    if not emails.empty:
                        callback(folder, emails, recache=True)
                    continue

                # Get current UIDs
                self.mail.select_folder(folder)
                current_uids = set(self.mail.search(['ALL']))
                
                # Check for changes
                if folder not in folder_uids:
                    folder_uids[folder] = current_uids
                    continue
                    
                new_uids = current_uids - folder_uids[folder]
                removed_uids = folder_uids[folder] - current_uids
                
                if new_uids or removed_uids:
                    folder_uids[folder] = current_uids
                    if new_uids:
                        emails = self.get_mail(filter='all', folders=[folder], uids=list(new_uids), return_dataframe=True)
                        if not emails.empty:
                            callback(folder, emails)
            
            except Exception as e:
                print(f"Error polling folder {folder}: {e}")

    def get_mail(self, filter='unseen', folders=["INBOX"], uids=None, return_dataframe=True):
        emails = []
        for folder in tqdm(folders, desc="Processing Folders", position=0, leave=False):
            self.mail.select_folder(folder)
            if uids is not None:
                messages = uids
            else:
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
        # self.mail.expunge()
