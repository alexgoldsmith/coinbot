import scrapy
import os
import time
import datetime



class CoinRequestSpider(scrapy.Spider):
    name = "coin_request"
    start_urls = ['https://coinmarketcap.com/all/views/all/',]
    
  #Parse links to currency pages
    def parse(self, response):
        #Create SelectorList of links
        links = response.css('a.link-secondary::attr(href)').extract()
        date_range_string = 'historical-data/?start=20130409&end=20190409'
        for link in links:
            table_link = link + date_range_string #To do: replace with date_range_string
            yield response.follow(url = table_link, callback = self.parse_pages)
    
    #Parse currency historical data
    def parse_pages(self, response):
        if not os.path.exists("html_files"):
            os.mkdir("html_files")
        page = response.url.split("/")[-3]
        filename = 'coins-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)