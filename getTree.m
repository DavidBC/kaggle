function [tree] = getTree(fields, targets, attributes, indices, node, parent)
max_split = 40;
ig_cutoff = 0.0015; %0.0015

tree.attribute = 'null';
tree.split = 0;
tree.score = 0;
tree.left = 'null';
tree.right = 'null';
tree.error = 'null';


best_ig = -inf; 
best_attribute = 0; 
best_val = 0;

cur_fields = fields(indices{node},:);
tree.size = size(cur_fields,1);
cur_targets = targets(indices{node});
gdi = GDI(cur_targets);

for i = 1:size(fields,2)
    column = cur_fields(:,i);

    vals = unique(column);
    splits = 0.5*(vals(1:end-1) + vals(2:end));
    num_splits = length(splits);
    if num_splits > max_split
        split_ind = [1:max_split]*floor(num_splits/max_split);
        splits = splits(split_ind);
    end
    if numel(vals) < 2
        continue
    end

    bin_mat = double(repmat(column, [1 numel(splits)]) < repmat(splits', [numel(column) 1]));

    gdi_X = zeros(1, size(bin_mat,2));
    for j = 1:size(bin_mat,2)
        gdi_X(j) = GDI_X(cur_targets, bin_mat(:,j));
    end
    IG = gdi - gdi_X;
    
    [val, ind] = max(IG);
    if val > best_ig
        best_ig = val;
        best_attribute = i;
        best_val = splits(ind);
    end
end
if best_ig < ig_cutoff
    return;
end

feat = cur_fields(:,best_attribute);
feat = feat < best_val;
indices = [indices; indices{node}(feat); indices{node}(~feat)];
indices{node} = [];
parent = [parent; node; node];

tree.attribute = best_attribute;
tree.split=best_val;
t = tabulate(cur_targets);
p = t(:,3) / 100;
tree.score = p(2);

n = numel(parent)-2;
tree.left = getTree(fields, targets, attributes, indices, n+1, parent);
tree.right = getTree(fields, targets, attributes, indices, n+2, parent);