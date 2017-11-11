# -*- coding: utf-8 -*- 
import sys
import tensorflow as tf
import configparser

def main():
    conf_file = '../conf/config.txt'
    config = configparser.ConfigParser()
    config.sections()
    config.read(conf_file)
    image_conf = config['images']
    print(image_conf['IMAGE_SIZE'])

if __name__ == '__main__':
    main()