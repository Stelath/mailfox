import os
import imaplib
import datetime
import email
from email.header import decode_header

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

# get uids
result, data = mail.uid('search', None, "ALL")
uids = data[0].split()
# get the last 500 emails
uids = uids[-500:]

for uid in uids:
    # fetch the email body (RFC822) for the given ID
    result, email_data = mail.uid('fetch', uid, '(BODY.PEEK[HEADER])')
    raw_email = email_data[0][1].decode("utf-8")
    email_message = email.message_from_string(raw_email)

    # header details
    date_tuple = email.utils.parsedate_tz(email_message['Date'])
    if date_tuple:
        local_date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
        local_message_date = "%s" %(str(local_date.strftime("%a, %d %b %Y %H:%M:%S")))
    email_from = str(decode_header(email_message['From'])[0][0])
    email_to = str(decode_header(email_message['To'])[0][0])
    subject = str(decode_header(email_message['Subject'])[0][0])

    print("From: ", email_from)
    print("To: ", email_to)
    print("Date: ", local_message_date)
    print("Subject: ", subject)
    print(" ")