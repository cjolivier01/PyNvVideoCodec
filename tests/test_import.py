import importlib


def test_pynvvideocodec_imports():
    """
    Basic smoke test to ensure the package can be imported.

    This verifies that the extension module and Python package are
    built and discoverable by the current Python environment.
    """
    module = importlib.import_module("PyNvVideoCodec")
    # Sanity checks on a couple of expected attributes
    assert hasattr(module, "__version__")
    assert hasattr(module, "SimpleDecoder")

