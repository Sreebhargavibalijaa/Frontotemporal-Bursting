data = readtable("filename.csv");
test_data = data(data.freq_band == "beta", :);

% GLME
model = fitglme(test_data, "br ~ period * region + (1|subject)");
anova(model)

% ANOVAN
[p, tbl, stats, terms] = anovan(test_data.br, ...
    {test_data.period, test_data.region}, ...
    'model', 'interaction', ...
    'varnames', {'test_data.period', 'test_data.region'});

% Tukey
[results, ~, ~, gnames] = multcompare(stats, 'Dimension', [1, 2]);

tbl = array2table(results,"VariableNames", ...
    ["test_data.period","test_data.region", ...
    "Lower Limit","A-B","Upper Limit","P-value"]);
tbl.("test_data.period") = gnames(tbl.("test_data.period"));
tbl.("test_data.region") = gnames(tbl.("test_data.region"));
