<h1 align="center">
<img src="assets/banner.png" width="300">
</h1><br>

![GitHub Release](https://img.shields.io/github/v/release/stelath/mailfox)
![GitHub License](https://img.shields.io/github/license/stelath/mailfox)

Mailfox is a simple and fast ai powered email toolkit to enhance you email workflow.

## Getting Started
### Installation
Run `pip install mailfox` to get started!

### Set Up
To setup mailfox you can simply run the `mailfox init` command which will launch you into a guided setup wizard:

The wizard will guiide you through the following steps:
* **Setting Credentials**: You'll be prompted to enter your email address and password. Optionally, you can enter an API key if you plan to use the LLM classifier.

* **Configuring Paths**: You'll be prompted to set paths for the email database and clustering data. You can also specify any flagged folders and set the default classifier (either "clustering" or "llm").

* **Downloading Emails**: You'll be asked if you'd like to download your emails to the VectorDB immediately.



## Features
### Classification

Classify emails into categories such as `Social`, `Promotions`, `Updates`, `Forums`, `Purchases`, `Spam` and `Others`.

When the `mailfox run` command is executed, it will classify all unread emails in your inbox and move them to the corresponding folder. It will then sleep for 5 minutes and repeat the process.

### Search

Coming Soon...