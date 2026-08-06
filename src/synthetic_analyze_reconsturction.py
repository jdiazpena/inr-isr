"""Compatibility alias for the historical misspelled log-error analysis name.

New code should import :mod:`synthetic_analyze_reconstruction_log_errors`.
"""

from __future__ import annotations

if __name__ == "__main__":
    from synthetic_analyze_reconstruction_log_errors import main

    main()
else:
    import sys
    import synthetic_analyze_reconstruction_log_errors as _implementation

    sys.modules[__name__] = _implementation
