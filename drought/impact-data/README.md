# Drought impact data

Script can be found under the folder of flood [impact-data](./impact-data/). 

Convenience script(s) to get impact data and map it to IBF-system format.

Supported data sources (all included by default):
 1. [DesInventar](https://www.desinventar.net/)
 2. [EM-DAT](https://www.emdat.be/)

```
Usage: get_impact_data.py [OPTIONS]

Options:
  --country TEXT   country (e.g. uganda)
  --disaster TEXT  type of disaster (e.g. drought)
  --help           Show this message and exit.
```
Example:
``` python get_impact_data.py --country=Uganda --disaster=drought```

Contact/support: Jacopo Margutti [jmargutti@redcross.nl](mailto:jmargutti@redcross.nl)
