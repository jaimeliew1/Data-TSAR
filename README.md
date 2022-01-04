# Data-TSAR
Data-TSAR (Data Time-series Satistical AggRegator) helps you to post-process time-series data files in bulk. Features include:
- Easy and readable workflow
- Parallel processing of data files
- Minimizes the number of disk read operations
- Can read HAWC2 and FLEX file formats
- Extraction of parameter values from file names
- Works with nested file structures

# Usage
Data-TSAR performs data aggregation on a specified file directory on files which match a simplified Regex pattern. Lets look at an example which calculates the mean and damage equivalent loads of some time series files and prints a Pandas DataFrame of the results:
```python
from TSAR import TimeSeriesAggregator

channels = {
    "channel1": 1,
    "channel2": 2,
}

agg = TimeSeriesAggregator("fictitious_data_dir", 
                        pattern="data_foo{foo}_bar{bar}.sel", 
                        channels=channels)

agg.add("mean", ["channel1", "channel2"])
agg.add("DEL", ["channel2"], m=4, neq=600)

agg.run_par()
df = agg.to_dataframe()

print(df)
```

First, a `TimeSeriesAggregator` object is created. It takes the following parameters:
- the path to the directory containing the data (in this case, `fictitious_data_dir`)
- The file pattern to match. Capture groups are indicated as curly brackets. For example, a file with the name `data_foo10_bar123.4.sel` will be matched, and the capture groups `foo = 10` and `bar=123.4` will be captured.
- The channel numbers to consider, and their corresponding labels to use. In this case, the label `channel1` corresponds to channel number `1`.

Next, the desired time series operations are lazily added to the `TimeSeriesAggregator` (no operations are performed until the `run` or `run_par` functions are called). The first argument is a string with the desired operation. Available operations include `mean`, `std`, `var`, `min`, `max`, `DEL`, `count`. The second argument is a list of channel labels to perform the operation. In this example, the mean of channel1 and channel2 are calculated, and the DEL of channel2 is calculated.

Finally, the parallelized calculation is performed by calling `agg.run_par()`. Every operation added to the aggregator will be performed on all files which match the file pattern. Data-TSAR is optimized to minimizes the number of disk read operations.

The data is then converted to a Pandas DataFrame and printed:
```
[1000 rows x 5 columns]
     foo  bar   channel1_mean  channel2_mean  channel2_DEL
0    10  1234.4 1.484160       2.177548       0.56382
1    20  1234.4 1.411251       2.009189       0.56382
2    30  1234.4 1.745251       2.224972       0.56382
3    10  2345.6 1.442648       2.461884       0.56382
4    20  2345.6 1.731026       2.940186       0.56382
..        ...       ...          ...            ...
995  20  2345.6 1.537893       2.218454       0.56382
996  30  2345.6 1.667881       2.812031       0.56382
997  10  9999.9 1.749854       2.666937       0.56382
998  20  9999.9 1.489747       2.320878       0.56382
999  30  9999.9 1.743557       2.678526       0.56382
```
As the aggregated data is stored in long-format, further postprocessing can be performed using pivot tables.

# Installation
Clone the repository and pip install:
```
git clone git@github.com:jaimeliew1/Data-TSAR.git
cd Data-TSAR
pip install .
```
