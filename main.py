# ─────────────────────────────────────────────
#  main.py  –  Entry point
# ─────────────────────────────────────────────
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.app import CarConfigApp

if __name__ == "__main__":
    app = CarConfigApp()
    app.mainloop()