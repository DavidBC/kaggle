function [classificationTree, attributes] = TreeClassifier(data_table, test)

M = table2array(data_table);
[n, m] = size(M);
field_table = data_table;
field_table(:,{'ID','TARGET'}) = [];

A = field_table.Properties.VariableNames;
[attributes] = normalize(M, A);
targets = data_table.TARGET;

if test == 1
    cvp = cvpartition(targets, 'Holdout', 0.2);

    train_fields = table2array(data_table(cvp.training, attributes));

    train_targets = targets(cvp.training, :);
else
    train_fields = table2array(data_table(:, attributes));
    train_targets = targets;
end

indices = {1:size(train_fields,1)};
classificationTree = getTree(train_fields, train_targets, attributes, indices, 1, 0);

if test == 1
    test_fields = data_table(cvp.test, attributes);
    test_targets = targets(cvp.test, :);

    prediction_scores = predictTree(classificationTree, test_fields);

    [~, ~, ~, AUC] = perfcurve(test_targets, prediction_scores, '1');
    disp(num2str(AUC));
end

