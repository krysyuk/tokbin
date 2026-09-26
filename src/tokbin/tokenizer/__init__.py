"""Tokenizers behind a small protocol (spec 10.3).

Only :mod:`tokbin.tokenizer.hf` touches the ``tokenizers`` package, and only lazily.
Reading code never imports anything from this package.
"""
