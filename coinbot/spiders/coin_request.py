import scrapy
import os
import time
import datetime

class CoinRequestSpider(scrapy.Spider):
    name = "coin_request"
    custom_settings = {
        'DOWNLOAD_DELAY': '10',
    }
    start_urls = ['https://coinmarketcap.com/all/views/all/',]
    
  #Parse links to currency pages
    def parse(self, response):
        #Create SelectorList of links
        links = response.css('a.currency-name-container::attr(href)').extract()
        date_range_string = 'historical-data/?start=20130409&end=20190409'
        for link in reversed(links):
            table_link = link + date_range_string 
            yield response.follow(url = table_link, callback = self.parse_pages)
    
    #Parse currency historical data
    def parse_pages(self, response):
        path = 'D:/Users/Alex/Git_Repositories/coinbot/html_files'
        if os.path.exists(path):
            os.chdir(path)
        else:
            os.mkdir(path)
        page = response.url.split("/")[-3]
        filename = 'coins-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)