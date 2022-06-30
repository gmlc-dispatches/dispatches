from _pytest.config import Config


_MARKERS = {
    'unit': 'quick tests that do not require a solver, must run in < 2 s',
    'component': 'quick tests that may require a solver',
    'integration': 'long duration tests',
}


def pytest_configure(config: Config):

    for name, descr in _MARKERS.items():
        config.addinivalue_line(
            'markers', f'{name}: {descr}'
        )
