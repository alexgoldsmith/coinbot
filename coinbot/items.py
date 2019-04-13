# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class CoinItem(scrapy.Item):
    # define the fields for your item here like:
    long_name = scrapy.Field()
    short_name = scrapy.Field()
    market_rank = scrapy.Field()
    date = scrapy.Field()
    day_open = scrapy.Field()
    day_high = scrapy.Field()
    day_low = scrapy.Field()
    day_close = scrapy.Field()
    day_volume = scrapy.Field()
    day_market_cap = scrapy.Field()

