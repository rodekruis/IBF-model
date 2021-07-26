# Python script for skill analysis of Model 2 - Observed precipitation

## `m2_skillsanalysis.py`

The script is to implement XGBoost to assess drought prediction skill of historical observed rainfall when negative crop yield anomaly is a drought impact proxy. As a results, it calculates a series of skill scores (POD, FAR, Precision, Recall, F1, etc). The skill analysis is performed at lead times 7-1 month to April.

The XGBoost parameters were already tuned, please don't change it unless you replicate the models for another countries.

The outputs per indicator are plots for the scores as well as feature importance. The latter indicates rainfall of which month influences more to the predictions at each lead time.


### Data input:

The script is now for Zimbabwe, data is available in FBF Zimbabwe channel on Teams.

For new indicators, the expected format is similar to one of DMP: in csv, with columns:
- `adm`: ZWE adm2 shapefile.
- `crop_anomaly`: ZWE crop anomaly.
- `precip_obs`: historical observed precipitation per ZWE adm2.


### Usage:

To easily run it with the data in the Zimbabwe channel, copy the script to `\4. FBA Research component for drought\Model 2 (precipitation)\script` in the channel and set console working directory.