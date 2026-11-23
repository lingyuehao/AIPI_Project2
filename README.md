# Why is my flight late?

An interactive explainer for a flight delay model, built on all 2023 domestic departures from New York (JFK, LGA, EWR).

**Live site:** https://lingyuehao.github.io/AIPI_Project2/

## What this is

Flight apps sometimes show a delay estimate next to a flight, but you never see where that number comes from. I wanted to build one of these predictors from public data and then try to explain it properly, and also check where it fails.

You can set up a flight (airline, airport, destination, date, time, weather) and get the chance it arrives 15+ minutes late. Then the page breaks that number down into how much each factor contributed. It also has:

- **The data**: delay rates by hour, month, airline, airport and weather before any model is involved
- **Explain a flight**: per-flight contributions (waterfall chart), plus the full learned curve for each factor
- **What-if**: searches for the smallest changes (time, day, airline, airport) that bring the risk below a target
- **Replay real flights**: real held-out 2023 flights, what the model said vs what actually happened, and the EBM explanation next to a TreeSHAP explanation of a black-box model
- **Audit**: calibration by airline / airport / time of day / season / weather, warning thresholds, severe delays, and a distribution shift test
- **Quiz**: a few statements to test whether you're reading the explanations correctly

The project mostly connects to interpretable ML / XAI, human-AI interaction (how a forecast is shown to a traveler) and AI evaluation (who the model works badly for).

## How to view it

Just open the live site above. Or download `index.html` and open it in a browser. Everything (data, photos) is inside that one file so it works offline. Fonts load from Google Fonts but it falls back to system fonts without internet.

Tips: arrow keys / page up / page down jump between sections, and the pill in the bottom left shows which chapter you're in.

## How to reproduce the numbers

You only need this if you want to rebuild the model and data behind the page.

```bash
pip install -r requirements.txt
python src/00_download_data.py     # gets the raw .rda files into data/raw/
python src/01_train_models.py      # features + EBM and LightGBM models, saves data/state.pkl
python src/02_export_for_site.py   # writes site_data/data.json
python src/03_embed_data.py        # puts the new json into index.html
```

Run them from the repo root. Steps 1 and 2 take a couple of minutes each on a laptop. I used Python 3.13, but anything recent should be fine.

### Files

```
index.html                 the whole site (html + css + js, data and photos embedded)
site_data/data.json        the exported data the page uses (same as what's embedded)
src/00_download_data.py    downloads the dataset
src/01_train_models.py     cleaning, features, training
src/02_export_for_site.py  exports model tables, audit numbers, sample flights
src/03_embed_data.py       embeds data.json into index.html
requirements.txt
```

### Data

- U.S. DOT Bureau of Transportation Statistics, Reporting Carrier On-Time Performance (2023), plus hourly NOAA weather at the departure airport
- I got it through the [nycflights23](https://github.com/moderndive/nycflights23) package, which has the BTS data already filtered to EWR/JFK/LGA. The download script pulls the `.rda` files from that repo. I tried to get it from transtats.bts.gov directly but couldn't, and this is the same data.
- 435,352 scheduled flights. Cancelled (10,738) and diverted (1,796) ones are dropped, leaving 422,818.

## Design decisions

**Target.** Arrival delay of 15+ minutes, because that's how BTS defines a late flight. So the model answers "if it flies, will it be late", not "will I get there".

**Two feature sets.** "At booking" uses airline, origin, destination (top 40, rest pooled as Other), month, day of week, scheduled hour, and airport busyness (number of flights scheduled out of the same airport in the same hour). "Day of departure" adds visibility, wind, precipitation and temperature at the origin. I wanted to show how the explanation changes once you know more.

**EBM as the main model.** An Explainable Boosting Machine is additive, one learned curve per feature with no interaction terms, so the explanation shown for a flight is exactly what the model computed and not an approximation. The page re-implements the EBM lookup tables in JavaScript, and `02_export_for_site.py` checks that this matches the Python model (max difference ~1e-16).

**LightGBM for comparison.** Same features, explained with TreeSHAP. Test AUC is about the same for both (EBM 0.719 vs LightGBM 0.725 for the day-of model), so here the glass box doesn't cost much accuracy. The interesting part is that the two explanations only agree on the top factor for 61% of flights.

**Train/test split by day.** 80/20, but whole calendar days go to one side, so the same day's weather and disruptions can't show up in both.

**What-if search.** It tries every single change and pairs of changes among the factors you allow, and only keeps airline/airport combos that actually flew that route in 2023 (you can turn this off). Weather is never changed. The ranking (fewer changes first, then smaller changes) and the easy / may cost / rarely possible tags are my own choices.

**Replay sample.** 320 test flights, but I oversampled late ones (90 flights 1h+ late) so there are enough misses to look at. The departure board on the page uses these too, so it looks later than reality. There's a note about this on the page.

**Distribution shift test.** A separate EBM trained only on Jan–Jun without the month feature, then tested on Jul–Dec. It's a deliberately simple setup to show the idea, not how the main model performs.

**Presentation.** Probabilities are shown with a 100-dot icon array and compared to the average flight, and there's a "guess first" mode before revealing the answer. The background photos are ones I took myself at different airports.

## Known limitations

- The model can't see the inbound aircraft (the data only has departures from NYC), destination weather, ground stops, crew issues or maintenance, which are probably some of the biggest causes of delay. AUC is only around 0.72.
- Cancellations aren't modeled at all.
- One region and one year. Nothing here should be assumed to transfer to other airports or years.
- Some categories are small. Hawaiian has only 362 flights and gets the biggest airline effect, which I wouldn't trust.
- Airline effects mix in route networks and schedules, so they shouldn't be read as an airline ranking.
- Several explanations on the page (summer thunderstorms in July, delays carrying over through the day, the model over-warning in bad weather because of ground programs) are my interpretations of the patterns. The data doesn't directly prove them.
- The weather presets (Clear / Rain / Storm / Snow) are example inputs I picked, not real days.
- Counterfactuals show how sensitive the model is, not actual advice. They can also make delay sound like the traveler's fault.
- Calibration is good overall but worse for some groups, e.g. it over-predicts in bad weather (about 32% predicted vs 26% actual) and gives flights that end up 3+ hours late a median of only ~32%.
