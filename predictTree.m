function [predictions] = predictTree(tree, test_fields)
n = height(test_fields);
test_fields = table2array(test_fields);

predictions = zeros(n, 1);
for i = 1:n
    travel = 1;
    cur_tree = tree;

    while travel == 1
       feature = test_fields(i, cur_tree.attribute);
       if feature <= cur_tree.split
           next_node = cur_tree.left;
       else
           next_node = cur_tree.right;
       end
       if strcmp(next_node.attribute,'null')==1 
           predictions(i) = cur_tree.score;
           travel = 0;
       else
           cur_tree = next_node;
       end
    end
end