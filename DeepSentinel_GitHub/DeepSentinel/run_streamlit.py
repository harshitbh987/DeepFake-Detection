#!/usr/bin/env python3
"""
run_streamlit.py — Launch DeepSentinel Streamlit interface
Usage:
    python run_streamlit.py
    python run_streamlit.py --port 8501
"""
import argparse, os, sys, subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8501)
    args = parser.parse_args()

    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')
    if not os.path.exists(app):
        print("ERROR: app.py not found."); sys.exit(1)

    print(f"Starting Streamlit on http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app,
                    '--server.port', str(args.port),
                    '--server.headless', 'true',
                    '--browser.gatherUsageStats', 'false'])

if __name__ == '__main__':
    main()
