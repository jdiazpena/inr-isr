"""Compatibility alias for the historical misspelled analysis module name.

New code should import :mod:`synthetic_analyze_reconstruction`.
"""

from __future__ import annotations

if __name__ == "__main__":
    from synthetic_analyze_reconstruction import main

    main()
else:
    import sys
    import synthetic_analyze_reconstruction as _implementation

    sys.modules[__name__] = _implementation
