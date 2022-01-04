from TSAR import TimeSeriesAggregator

res_dir = "fictitious_data"
file_pattern = "testdata_foo{foo}_bar{bar}.sel"
channels = {
    "channel1": 1,
    "channel2": 2,
}

if __name__ == "__main__":

    # Define a time series aggregator based on a result directory (res_dir), a
    # file name pattern with capture groups (file_pattern) and a dictionary of
    # channel indices as label-index pairs (channels).
    agg = TimeSeriesAggregator(res_dir, pattern=file_pattern, channels=channels)

    # add operation to calculate the mean of both channels.
    agg.add("mean", ["channel1", "channel2"])
    # add operation to calculate the standard deviation of channel1.
    agg.add("std", ["channel1"])
    # add operation to calculate the damage equivalent load (DEL) of channel2.
    agg.add("DEL", ["channel2"], m=4, neq=600)

    # Perform the data processing in parallel
    agg.run_par()

    # convert results to dataframe
    df = agg.to_dataframe()

    print(df)
