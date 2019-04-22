# Coinbot

This repository contains code for scraping historical data from coinmarketcap.com as well as a notebook analysis of the scraped data. 

## Prerequisites

This project requires both Scrapy and JupyterLab.

Installation instructions for Scrapy: 
http://doc.scrapy.org/en/latest/intro/install.html

Installation instructions for JupyterLab
https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html

## User Guide

To download the html files required for dataset construction, navigate to the coinbot directory and enter the following command:

```
scrapy crawl coin_request
```

To recreate the dataset used for the analysis, enter the following command from the coinbot directory after downloading the required html files (note: in order to run properly, the file path to access the html files must be editted in coin_parse.py)

```
scrapy crawl coin_parse
```

If you wish to skip straight to the analysis, all necessary files are included in the repository. Simply open up crypto_notebook.ipynb and run all cells!