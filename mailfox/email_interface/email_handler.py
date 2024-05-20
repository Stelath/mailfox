import os
import imaplib
import datetime
import multiprocessing
import email
from email.header import decode_header
from tqdm.auto import tqdm
import pandas as pd

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
    
    def get_all_folders(self):
        result, data = self.mail.list()
        return [d.decode('utf-8').split(' "/" ')[1][1:-1] for d in data]
    
    def get_folder_uids(self, folder):
        result, data = self.mail.select(folder)
        if result == 'OK':
            result, data = self.mail.uid('search', None, "ALL")
            return [d.decode('utf-8') for d in data[0].split()]
        else:
            print(f"Failed to select folder {folder}")
            return []
    
    def get_mail(self, filter='unseen', *, uids=None, folders=["INBOX"], return_dataframe=True):
        emails = []
        
        if folders != [] and folders is not None:
            uids = []
            for folder in folders:
                result, data = self.mail.select(folder)
                if result == 'OK':
                    if filter == 'unseen':
                        result, data = self.mail.uid('search', None, "(UNSEEN)")
                        uids += data[0].split()
                    elif filter == 'seen':
                        result, data = self.mail.uid('search', None, "(SEEN)")
                        uids += data[0].split()
                    elif filter == 'all':
                        result, data = self.mail.uid('search', None, "ALL")
                        uids += data[0].split()
                    elif filter == 'uids' and uids is not None:
                        result = 'OK'
                        uids += [uid.encode('utf-8') for uid in uids]
                    else :
                        print("Invalid filter. Please use 'unseen', 'all', or 'uids'.")
                        return
                else:
                    print(f"Failed to select folder {folder}")
                    continue
        else:    
            if filter == 'unseen':
                result, data = self.mail.uid('search', None, "(UNSEEN)") # search and return uids of unseen emails
                uids = data[0].split()
            elif filter == 'seen':
                result, data = self.mail.uid('search', None, "(SEEN)")
                uids = data[0].split()
            elif filter == 'all':
                result, data = self.mail.uid('search', None, "ALL")
                uids = data[0].split()
            elif filter == 'uids' and uids is not None:
                result = 'OK'
                uids = [uid.encode('utf-8') for uid in uids]
            else:
                print("Invalid filter. Please use 'unseen', 'all', or 'uids'.")
                return
        
        if uids != [] and uids is not None:
            for num in tqdm(uids, desc="Processing Emails", leave=False):
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
                    folder = str(decode_header(email_message['Folder'])[0][0])
                    id = num.decode('utf-8')
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

                # return the email details
                emails.append({'id': id, 'from': email_from, 'to': email_to, 'subject': subject, 'date': local_message_date, 'body': body})
        else:
            print("No new emails to read.")
        
        if return_dataframe:
            return pd.DataFrame(emails)
        
        return emails
    
    def create_folder(self, folder):
        result = self.mail.create(folder)
        if result[0] == 'OK':
            print(f"Created folder {folder}")
        else:
            print(f"Failed to create folder {folder}")
    
    def move_mail(self, uids, folder):
        for uid in uids:
            result = self.mail.uid('COPY', uid, folder)
            if result[0] == 'OK':
                mov, data = self.mail.uid('STORE', uid , '+FLAGS', '(\Deleted)')
                self.mail.expunge()
                print(f"Moved {uid} to {folder}")
            else:
                print(f"Failed to move {uid} to {folder}")
    
    def delete_mail(self, uids):
        for uid in uids:
            mov, data = self.mail.uid('STORE', uid , '+FLAGS', '(\Deleted)')
            self.mail.expunge()
            print(f"Deleted {uid}")