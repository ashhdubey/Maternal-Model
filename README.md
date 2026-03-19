# Maternal-Model

### Data Collection Logic
- Basic maternal profile information (age, pregnancy stage)
- Daily lifestyle indicators such as sleep duration, diet quality, physical activity, and stress level
- Self-reported symptoms and general discomfort indicators


### Missing Data Handling Strategy
- If data is missing for one or two days, the system uses previous valid entries to maintain continuity.
- For longer gaps, the system flags the data as incomplete and reduces confidence in trend analysis.
- No predictions are generated when critical data fields are missing.
- The user interface gently reminds users to submit daily inputs without forcing data entry.

### Selection of Machine Learning Approach
- Type of ML Used = Regression + Trend Analysis
- Input → last N days health data
- Output → today’s health score or trend direction
