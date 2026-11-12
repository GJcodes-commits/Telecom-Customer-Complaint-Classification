# Dataset

The project downloads a filtered sample from the official FCC Consumer Complaints dataset when you run `python download_data.py`.

Source: https://opendata.fcc.gov/Consumer/CGB-Consumer-Complaints-Data/3xyp-aqkj

The downloader requests up to 50,000 recent records for Phone, Internet and TV complaint forms and retains:
- `ticket_created`: complaint submission timestamp
- `issue_type`: service/form category
- `method`: how the consumer obtains the service
- `issue`: specific complaint issue (the prediction target)
- `state`: consumer's state

The FCC notes that complaints contain information selected by consumers and that it does not verify the allegations. These records are not a representative sample of all telecom customers.

The CSV is intentionally not committed to GitHub. Run the downloader locally to retrieve it.
