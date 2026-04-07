# Type stubs for NautilusTrader

<img width="1792" height="846" alt="image" src="https://github.com/user-attachments/assets/efbaadb8-0103-4b74-bc00-fac456e95e57" />


`nautilus-trader-cython-stubs` provides **`.pyi` type stubs** for the [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) trading platform, specifically for its **Cython interface**.

These stubs serve the following purposes:

- Enhance **IntelliSense and docstring support** in **Visual Studio Code** (via Pylance) and other IDEs  
- Provide consistent **type annotations** and **docstring visibility** for Cython modules  
- Enable proper **import resolution** for Cython-based APIs 

## Generation

Generate `.pyi` stub files from NautilusTrader's `.pyx` Cython sources:

```bash
# One-command workflow: clean -> generate -> validate
bash generate_stubs.sh

# Or step-by-step
python scripts/stub_generator.py --all
bash scripts/validate_stubs.sh
```

**Requirements**: Python 3.11+ and Cython

## Installation

Copy generated stubs to your installed `nautilus_trader` package:

```bash
rsync -a ./stubs/ {path/to/python/site-packages}/nautilus_trader
```

Example:
```bash
rsync -a ./stubs/ ~/.local/lib/python3.14/site-packages/nautilus_trader
```

## Limitations

- These stubs cover only the Cython interface — Rust bindings are not included.
- If you find any missing or inconsistent APIs, [open an issue](https://github.com/woung717/nautilus-trader-cython-stubs/issues).

