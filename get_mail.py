import os
import imaplib
import datetime
import email
from email.header import decode_header
from tqdm.auto import tqdm
import pandas as pd

# Read credentials from txt file
with open('credentials.txt') as f:
    lines = f.readlines()
    username = lines[0].strip()
    password = lines[1].strip()

# create an IMAP4 class with SSL 
mail = imaplib.IMAP4_SSL("imap.gmail.com")
# authenticate
mail.login(username, password)

# select the mailbox you want to delete in
# if you want SPAM, use "INBOX.SPAM"
mailbox = "INBOX"
mail.select(mailbox)

emails = []

# get uids
result, data = mail.uid('search', None, "ALL")
uids = data[0].split()
# get the last 1000 emails
uids = uids[-5000:]
failed = 0

for uid in tqdm(uids):
    # try:
    # fetch the email body (RFC822) for the given ID
    result, email_data = mail.uid('fetch', uid, '(BODY.PEEK[])')
    raw_email = email_data[0][1].decode("latin-1")
    email_message = email.message_from_string(raw_email)
    
    # header details
    date_tuple = email.utils.parsedate_tz(email_message['Date'])
    if date_tuple:
        local_date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
        local_message_date = "%s" %(str(local_date.strftime("%a, %d %b %Y %H:%M:%S")))
    email_from = str(decode_header(email_message['From'])[0][0])
    email_to = str(decode_header(email_message['To'])[0][0])
    subject = str(decode_header(email_message['Subject'])[0][0])
    
    body = ""
    if email_message.is_multipart():
        for part in email_message.get_payload():
            if part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True).decode("latin-1")
            # elif part.get_content_type() == 'text/html':
            #     body += part.get_payload(decode=True).decode()
    else:
        body = email_message.get_payload(decode=True).decode("latin-1")
    
    # print(subject)
    # print(body)
    # print('\n')
        
        
    # except:
        # print(f"Error occurred while processing email with uid: {uid}")
        # failed += 1
        # pass
    
    # create a dictionary to store email details
    email_dict = {
        "Date": local_message_date,
        "From": email_from,
        "To": email_to,
        "Subject": subject,
        "Body": body
    }
    emails.append(email_dict)


# convert the dictionary to a pandas DataFrame
df = pd.DataFrame(emails)

# write the DataFrame to a parquet file
df.to_parquet('emails.parquet', engine='pyarrow')

# print(f"Finished!\nFailed to process {failed} emails.")
    