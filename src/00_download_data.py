# Downloads the 2023 NYC flights data from the nycflights23 package on GitHub.
# I couldn't pull straight from transtats.bts.gov so I'm using this copy, it's the
# same BTS on-time data (plus NOAA weather) just already filtered to EWR/JFK/LGA.
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/moderndive/nycflights23/main/data/"
FILES = ["flights", "weather", "airlines", "airports", "planes"]

os.makedirs("data/raw", exist_ok=True)
for f in FILES:
    out = f"data/raw/{f}.rda"
    if os.path.exists(out):
        print("already have", out)
        continue
    print("downloading", f)
    urllib.request.urlretrieve(BASE + f + ".rda", out)
print("done")
