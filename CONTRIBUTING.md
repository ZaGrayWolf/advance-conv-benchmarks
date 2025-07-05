# Contributing to Conv Benchmarks

We welcome contributions to the Conv Benchmarks library! This document provides guidelines for contributing.

## Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/conv-benchmarks.git
cd conv-benchmarks
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e ".[dev]"
```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **pytest** for testing

Run these before submitting:
```bash
black conv_benchmarks tests examples
isort conv_benchmarks tests examples
flake8 conv_benchmarks tests examples
pytest tests/
```

## Adding New Convolution Types

To add a new convolution implementation:

1. **Create the layer file** in `conv_benchmarks/layers/`
2. **Implement the nn.Module** with a consistent interface:
   - `__init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs)`
   - `forward(self, x)` method
3. **Add imports** to `conv_benchmarks/layers/__init__.py`
4. **Update the benchmark** to include your new layer
5. **Add tests** in `tests/test_layers.py`

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

For coverage:
```bash
pytest tests/ --cov=conv_benchmarks
```

## Submitting Changes

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature-name`
3. **Make your changes** and add tests
4. **Run the test suite** and ensure all tests pass
5. **Submit a pull request** with a clear description

## Reporting Issues

Please use the GitHub issue tracker to report bugs or request features. Include:
- Python version
- PyTorch version
- Operating system
- Minimal code example demonstrating the issue

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
