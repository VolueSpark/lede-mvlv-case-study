from datetime import datetime
import os

from plotting import plot
from data import fetch

PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    days=90
    from_date = datetime(2024, 4, 1)

    fetch(from_date=from_date, days=days, path=PATH)
    plot(from_date=from_date, days=days, path=PATH)


