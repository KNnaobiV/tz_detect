# Timezone Detector

The Timezone Detector script is designed to detect the timezone information of timeseries data stored in CSV files. It determines the timezone of unaware timeseries data and set the timezone.

## Features
- **Detect Timezone**: Analyzes the datetime information of timeseries data to determine the timezone.
- **Set Timezone**: Sets the timezone of unaware timeseries data to the specified timezone.

## Requirements
- Python 3.x
- pandas
- numpy
- polars
- pytz

## Usage
1. Clone the repository or download the script file `timezone_detector.py`.
2. Ensure that your timeseries data is stored in CSV format.
3. Run the script using Python by providing the necessary arguments:


- To return the timezone of the unaware timeseries
```bash
python timezone_detector.py -g <file1.csv> <file2.csv>
```

- To set the timezone of the unaware timeseries
```bash
python timezone_detector.py -s <file1.csv> <file2.csv>
```