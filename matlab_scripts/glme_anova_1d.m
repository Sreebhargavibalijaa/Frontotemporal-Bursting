data = readtable("file.csv");
test_data = data(data.freq_band == "high_gamma", :);

[p, tbl, stats] = anova1(test_data.dur, test_data.period);
[results, ~, ~, gnames] = multcompare(stats);