clear;

filename = 'train.csv';
test_file = 'test.csv';

train_table = readtable(filename);

%tic
test = 0;
[classificationTree, attributes] = TreeClassifier(train_table, test);
%disp(num2str(toc));

testdata = readtable(test_file);
testdata_id = testdata.ID;
[n, m] = size(attributes);
testdata = testdata(:, attributes);
prediction_scores = predictTree(classificationTree, testdata);

out = [testdata_id, prediction_scores];

fid = fopen('results.csv', 'wt');
fprintf(fid, 'ID,Target\n');
fclose(fid);
dlmwrite('results.csv', out, '-append', 'precision', 10);