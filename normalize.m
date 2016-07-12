function [ attributes ] = normalize( fields, attributes )
    
    %remove constant columns
    [n, m] = size(fields);
    sd = std(fields);
    del_col = [];
    k = 0;
    for i = 1:m;
        if sd(i) == 0;
            k = k+1;
            del_col(k) = i;
        end
    end
    %disp(['constant columns: ', mat2str(del_col)]);
    if ~isempty(del_col);
        fields(:, del_col) = [];
        attributes(:, del_col) = [];
    end
    
    %remove duplicate columns
    [n, m] = size(fields);
    del_col = [];
    k = 0;
    for i = 1:m;
        compare_col = fields(:, i);
        for j = (i+1):m;
            if compare_col == fields(:, j);
                k = k+1;
                del_col(k) = j;
            end
        end
    end
    %disp(['duplicate columns: ', mat2str(del_col)]);
    if ~isempty(del_col);
        fields(:, del_col) = [];
        attributes(:, del_col) = [];
    end

    
    %[n, m] = size(fields);
    %for i = 1:m;
    %    fields(:, i) = fields(:, i)/ norm(fields(:, i));
    %end

end
