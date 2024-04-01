import os
import imaplib
import datetime
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
        return data[0].split()
    
    def get_new_mail(self, *, unseen=True, uids=None):
        emails = []
        
        if uids is not None:
            result = 'OK'
        else:
            if unseen:
                result, data = self.mail.uid('search', None, "(UNSEEN)") # search and return uids of unseen emails
            else:
                result, data = self.mail.uid('search', None, "ALL")
            
            uids = data[0].split()
        
        if result == 'OK':
            for num in uids:
                result, email_data = self.mail.uid('fetch', num, '(BODY.PEEK[])') # fetch the email body (peek = not mark as read)
                raw_email = email_data[0][1]
                raw_email_string = raw_email.decode('utf-8')
                email_message = email.message_from_string(raw_email_string)

                # get the email details
                date_tuple = email.utils.parsedate_tz(email_message['Date'])
                if date_tuple:
                    local_date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
                    local_message_date = "%s" %(str(local_date.strftime("%a, %d %b %Y %H:%M:%S")))
                email_from = str(decode_header(email_message['From'])[0][0])
                email_to = str(decode_header(email_message['To'])[0][0])
                subject = str(decode_header(email_message['Subject'])[0][0])
                id = num.decode('utf-8')
                
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
        
        return emails
    
    def add_email_to_group(self, email_id, label):
        result = self.mail.uid('STORE', email_id, '+X-GM-LABELS', label)
        if result[0] == 'OK':
            print(f"Email {email_id} has been added to {label}")
        else:
            print(f"Failed to add email {email_id} to {label}")
    
    def create_label(self, label):
        result = self.mail.create(label)
        if result[0] == 'OK':
            print(f"Label {label} has been created.")
        else:
            print(f"Failed to create label {label}")