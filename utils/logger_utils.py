#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging, sys

loggers = {}


def get_logger(name='default'):
    try:
        import loguru
        return loguru.logger
    except ImportError:
        pass

    global loggers
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        # formatter = logging.Formatter(
        #     fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        loggers[name] = logger
        return logger