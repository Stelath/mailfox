import os
import imaplib
import datetime
import multiprocessing
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
        
        # if you want SPAM, use "INBOX.SPAM"
        mailbox = "INBOX"
        self.mail.select(mailbox)
    
    def get_all_mail_uids(self):
        result, data = self.mail.uid('search', None, "ALL")
        return [d.decode('utf-8') for d in data[0].split()]
    
    def format_folders(self, folders, plain=False):
        formatted_folders = []
        
        if not plain:
            for folder in folders:
                if folder[0] != '"':
                    folder = '"' + folder + '"'
                    formatted_folders.append(folder)
                else:
                    formatted_folders.append(folder)
        else:
            for folder in folders:
                if folder[0] == '"':
                    folder = folder[1:-1]
                    formatted_folders.append(folder)
                else:
                    formatted_folders.append(folder)
                
        return formatted_folders
    
    def get_all_folders(self):
        result, data = self.mail.list()
        return [d.decode('utf-8').split(' "/" ')[1] for d in data]
    
    # FIXME: Will list subfolders of say "personal" as "personal/finance", but also (and incorrectly) as
    # "other_dir/personal", order doesn't matter to it
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
        
        if folders != [] and folders is not None:
            uids = {}
            for folder in folders:
                result, data = self.mail.select(folder)
                if result == 'OK':
                    if filter == 'unseen':
                        result, data = self.mail.uid('search', None, "(UNSEEN)")
                        uids[folder] = data[0].split()
                    elif filter == 'seen':
                        result, data = self.mail.uid('search', None, "(SEEN)")
                        uids[folder] = data[0].split()
                    elif filter == 'all':
                        result, data = self.mail.uid('search', None, "ALL")
                        uids[folder] = data[0].split()
                    elif filter == 'uids' and uids is not None:
                        result = 'OK'
                        uids[folder] = [uid.encode('utf-8') for uid in uids]
                    else :
                        print("Invalid filter. Please use 'unseen', 'all', or 'uids'.")
                        return
                else:
                    print(f"Failed to select folder {folder}")
                    continue
        else:
            print("No folders to read.")
            return
        
        if uids != [] and uids is not None:
            for folder, uids in tqdm(uids.items(), desc="Processing Folders", position=0, leave=False):
                self.mail.select(folder)
                for num in tqdm(uids, desc=f"Getting Emails from {folder}", position=1, leave=False):
                    result, email_data = self.mail.uid('fetch', num, '(BODY.PEEK[])') # fetch the email body (peek = not mark as read)
                    raw_email = email_data[0][1]
                    try:
                        raw_email_string = raw_email.decode('utf-8')
                    except:
                        continue
                    email_message = email.message_from_string(raw_email_string)

                    # get the email details
                    date_tuple = email.utils.parsedate_tz(email_message['Date'])
                    if date_tuple:
                        local_date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
                        local_message_date = "%s" %(str(local_date.strftime("%a, %d %b %Y %H:%M:%S")))
                        
                    try:
                        email_from = str(decode_header(email_message['From'])[0][0])
                        email_to = str(decode_header(email_message['To'])[0][0])
                        subject = str(decode_header(email_message['Subject'])[0][0])
                        uuid = self.hash_email({'from': email_from, 'to': email_to, 'subject': subject, 'date': local_message_date}) 
                        uid = num.decode('utf-8')
                    except:
                        continue
                    
                    body = ""
                    if email_message.is_multipart():
                        for part in email_message.get_payload():
                            if part.get_content_type() == 'text/plain':
                                body = part.get_payload(decode=True).decode("latin-1")
                            # elif part.get_content_type() == 'text/html':
                            #     body += part.get_payload(decode=True).decode()
                    else:
                        body = email_message.get_payload(decode=True).decode("latin-1")
                    
                    raw_body = body
                    
                    soup = BeautifulSoup(body, 'lxml')
                    body = soup.get_text()
                    body = self._strip_repeated_chars(body)
                    
                    # return the email details
                    emails.append({'uid': uid, 'folder': folder, 'uuid': uuid, 'from': email_from, 'to': email_to, 'subject': subject, 'date': local_message_date, 'body': body, 'raw_body': raw_body})
        else:
            print("No new emails to read.")
        
        if return_dataframe:
            return pd.DataFrame(emails)
        
        return emails
    
    def _strip_repeated_chars(self, s, chars = ['\n', '\t']):
        for char in chars:
            s = re.sub(f'{char}+', char, s)
        return s
    
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
                mov, data = self.mail.uid('STORE', uid , '+FLAGS', '(\Deleted)')
                self.mail.expunge()
            else:
                print(f"Failed to move {uid} to {folder}")
    
    def delete_mail(self, uids):
        folder = "INBOX.Trash"
        self.move_mail(uids, folder)
        # for uid in uids:
        #     mov, data = self.mail.uid('STORE', uid , '+FLAGS', '(\Deleted)')
        #     self.mail.expunge()
        #     print(f"Deleted {uid}")