"""
Poor Man's Configurator. Probably a terrible idea, but very convenient.

Usage from train.py / sample.py:
    exec(open('configurator.py').read())

It reads sys.argv and mutates the caller's globals() in place:
  - a bare token (no '='): treated as a python config file to exec, e.g.
        python train.py config/train_shakespeare_char.py
  - a --key=value token: overrides the global `key` with the parsed `value`, e.g.
        python train.py --batch_size=32 --compile=False

Values are parsed with ast.literal_eval so numbers/bools/lists come through with
the right type; anything that fails to parse is kept as a plain string. The new
value's type must match the existing global's type (a small safety check).
"""
import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # a config file to execute in the caller's namespace
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # a --key=value command-line override
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)  # e.g. "32" -> 32, "False" -> False
            except (SyntaxError, ValueError):
                attempt = val                # leave it as a string
            assert type(attempt) == type(globals()[key])
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
