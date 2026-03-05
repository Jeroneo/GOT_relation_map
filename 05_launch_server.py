#!/usr/bin/env python3
"""
serve.py — Local server for index.html
Serves the current directory on http://localhost and opens the browser.
"""

import http.server
import socketserver
import os
import sys

PORT = 80
FILE = "index.html"

def run():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None  # silence request logs

    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n  Game of Thrones — Character Map")
        print(f"  Serving on http://localhost:{PORT}")
        print(f"  Press Ctrl+C to stop\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
            sys.exit(0)

if __name__ == "__main__":
    run()
