import scrapy
import os
import glob
from coinbot.items import CoinItem

class CoinParseSpider(scrapy.Spider):
    name = "coin_parse"
     custom_settings = {
        'CONCURRENT_REQUESTS' : '100',
    }

    def start_requests(self):
        path = 'D:/Users/Alex/Git_Repositories/coinbot/html_files/'
        os.chdir(path)
        file_list = glob.glob('*.html')
        for file in file_list:
            file_url = 'file:///' + path + file
            yield scrapy.Request(url=file_url, callback=self.parse)

    def parse(self, response):
        table_rows = response.xpath('//*[@id="historical-data"]/div/div[2]/table/tbody/*')
        for tr in table_rows:
            table_item = CoinItem()
            name_string = response.xpath('//*[@id="historical-data"]/div/h1/text()').get()
            string_split = name_string.split()
            table_item['long_name'] = string_split[-1]
            table_item['short_name'] = response.css('div.h1.details-panel-item--name').xpath('./span/text()').get()
            table_item['market_rank'] = response.xpath('/html/body/div[2]/div/div[1]/div[6]/div[2]/table/tbody/tr[3]/td/text()').get().strip()
            table_item['date'] = tr.xpath('./td[1]/text()').get()
            table_item['day_open'] = tr.xpath('./td[2]/text()').get()
            table_item['day_high'] = tr.xpath('./td[3]/text()').get()
            table_item['day_low'] =tr.xpath('./td[4]/text()').get()
            table_item['day_close'] = tr.xpath('./td[5]/text()').get()
            table_item['day_volume'] = tr.xpath('./td[6]/text()').get()
            table_item['day_market_cap'] = tr.xpath('./td[7]/text()').get()
            yield table_item