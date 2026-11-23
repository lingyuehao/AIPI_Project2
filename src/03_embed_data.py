# The page is a single html file (so it works offline / on github pages without fetch issues),
# so the json gets pasted into a <script type="application/json"> tag inside index.html.
# Run this after 02_export_for_site.py to update the numbers on the page.
import re

html = open("index.html", encoding="utf-8").read()
data = open("site_data/data.json", encoding="utf-8").read()
new_html, n = re.subn(r'(<script type="application/json" id="flightdata">).*?(</script>)',
                      lambda m: m.group(1) + data + m.group(2), html, count=1, flags=re.S)
assert n == 1, "couldn't find the data block in index.html"
open("index.html", "w", encoding="utf-8").write(new_html)
print("updated index.html")
