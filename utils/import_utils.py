#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib


def import_custom_func(script_path: str, func_name: str) -> callable:
    """
    A function that imports a custom function based on the provided configuration.
    Takes a dictionary `config` containing 'custom' information with 'script_path' and 'func_name'.
    Returns the imported custom function.
    """

    if script_path.startswith('/'):
        script_path = script_path[1:]
    if script_path.endswith('.py'):
        script_path = script_path[:-3]
    custom_module = importlib.import_module(script_path.replace('/', '.'))
    custom_func = getattr(custom_module, func_name)
    return custom_func
