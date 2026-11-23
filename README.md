# Why is my flight late?

Interactive site that explains a flight delay model trained on 2023 domestic departures from JFK, LGA and EWR.

Live site: https://lingyuehao.github.io/AIPI_Project2/

## About

Some flight apps show a delay estimate next to a flight, but you can't see how they came up with that. For this project I trained my own delay model on public data and built a page where you can look at why it predicts what it predicts, and where it gets things wrong.

On the page you pick a flight (airline, airports, month, day, time, and weather if you want) and it gives the probability of arriving 15+ min late. Then it shows how much each feature added or removed. The sections are:

- The data: late rates by hour, month, airline, airport and weather, no model yet
- Explain a flight: waterfall chart of feature contributions, and the curve the model learned for each feature
- What-if: finds small changes (time, day, airline, airport) that would get the risk under a target
- Replay: real test-set flights with the prediction vs what actually happened, plus EBM vs TreeSHAP explanations side by side
- Audit: calibration by group, warning cutoffs, severe delays, and a distribution shift check
- Quiz: a few statements about the results to see if you're reading them right

It mainly relates to interpretable ML / XAI, human-AI interaction and AI evaluation from the course.

## Viewing it

Open the live site. You can also download `index.html` and open it in any browser since the data and photos are embedded in the file. Fonts come from Google Fonts so without internet it uses default fonts.

Arrow keys and page up/down move between sections.

## Reproducing it

```bash
pip install -r requirements.txt
python src/00_download_data.py
python src/01_train_models.py
python src/02_export_for_site.py
python src/03_embed_data.py
```

Run from the repo root. `00` downloads the raw data to `data/raw/`, `01` builds features and trains the models, `02` writes `site_data/data.json`, and `03` copies that json into `index.html`. Training and export took around 2-3 min total on my machine. I used Python 3.13.

Files:

```
index.html                 the site (html/css/js, data and photos embedded)
site_data/data.json        exported data used by the page
src/00_download_data.py
src/01_train_models.py
src/02_export_for_site.py
src/03_embed_data.py
requirements.txt
```

Data: BTS Reporting Carrier On-Time Performance for 2023 plus hourly NOAA weather at the origin airport. I used the copy in the [nycflights23](https://github.com/moderndive/nycflights23) package because I couldn't download from transtats.bts.gov directly, and it's already filtered to the three NYC airports. There are 435,352 scheduled flights. I dropped 10,738 cancelled and 1,796 diverted flights, so 422,818 are left.

## Design choices

Target is arrival delay >= 15 min since that's the BTS definition of late. Cancelled flights aren't included, so the model is really "if it flies, will it be late".

I made two versions of the features. The booking version has airline, origin, destination (top 40 + Other), month, day of week, scheduled hour and how many flights are scheduled out of that airport in the same hour. The day-of version adds visibility, wind, precipitation and temperature at the origin. I wanted to compare the explanations with and without weather.

Main model is an Explainable Boosting Machine with no interaction terms, so each prediction is just the sum of one value per feature and the explanation is exact. The page computes predictions in JavaScript from the exported EBM tables. `02_export_for_site.py` checks these match the Python model (difference around 1e-16).

I also trained LightGBM on the same features and explained it with TreeSHAP. AUC is about the same (0.719 vs 0.725 for day-of), but the two only pick the same top feature for 61% of flights, which I thought was one of the more interesting results.

Train/test split is 80/20 by calendar day so a single day's weather or disruption can't end up in both.

For what-if I just try every single change and every pair of changes over the features you allow, and by default only keep airline/airport combos that actually flew the route in 2023. Weather is never changed. The ordering and the "easy / may cost / rarely possible" labels are my own call.

The replay set is 320 test flights but I oversampled late ones (90 that were 1h+ late) so there are enough misses to look at. The departure board at the top uses the same flights, so it looks worse than reality. There's a note about this on the page.

For the distribution shift part I trained a separate EBM on Jan-Jun only without the month feature and tested it on Jul-Dec. It's just meant to show the problem, it's not the main model.

The background photos are ones I took at different airports.

## Limitations

- The model doesn't see the inbound aircraft (data only has departures from NYC), destination weather, ground stops, crew or maintenance problems. Those are probably big causes of delay, and AUC is only ~0.72.
- No cancellations.
- Only NYC and only 2023.
- Some groups are tiny. Hawaiian has 362 flights and gets the biggest airline effect, I wouldn't trust that.
- Airline effects also reflect routes and schedules, so they aren't a ranking of airlines.
- Some explanations on the page (July thunderstorms, delays building up during the day, over-warning in bad weather because of ground programs) are my guesses about the patterns and aren't proven by the data.
- The Clear / Rain / Storm / Snow presets are values I picked, not real days.
- What-if results show what the model reacts to, not advice, and they can make delays sound like the passenger's fault.
- Calibration is fine overall but worse in some groups. In bad weather it predicts ~32% vs 26% actual, and flights that end up 3h+ late only get a median prediction of ~32%.
