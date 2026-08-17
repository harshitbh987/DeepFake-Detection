#!/usr/bin/env python3
"""
run_flask.py — Launch DeepSentinel Flask interface
Usage:
    python run_flask.py
    python run_flask.py --port 5000 --debug
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
try:
    import numpy._core.multiarray as _nca
    torch.serialization.add_safe_globals([_nca.scalar])
except Exception:
    pass
_orig = torch.load
def _safe(f, map_location=None, **kw):
    kw['weights_only'] = False
    return _orig(f, map_location=map_location, **kw)
torch.load = _safe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',  default='0.0.0.0')
    parser.add_argument('--port',  type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    flask_app = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flask_app.py')
    if not os.path.exists(flask_app):
        print("ERROR: flask_app.py not found."); sys.exit(1)

    print(f"Starting Flask on http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")

    import importlib.util
    spec = importlib.util.spec_from_file_location('flask_app', flask_app)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
