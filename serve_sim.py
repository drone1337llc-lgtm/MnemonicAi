#!/usr/bin/env python3
"""Serve the interactive simulator locally and open it in your browser.

    python3 serve_sim.py            # http://localhost:8000/simulator.html
    python3 serve_sim.py 9000       # choose a port

Pure standard library — no dependencies. The simulator is a self-contained HTML
file (simulator.html); it runs entirely in the browser.
"""
from __future__ import annotations

import http.server
import os
import socketserver
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = "simulator.html"


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.chdir(HERE)
    if not os.path.isfile(PAGE):
        print(f"! {PAGE} not found next to this script.")
        return 1
    handler = http.server.SimpleHTTPRequestHandler
    url = f"http://localhost:{port}/{PAGE}"
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving the MnemonicAi simulator at {url}")
        print("Press Ctrl+C to stop.")
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
