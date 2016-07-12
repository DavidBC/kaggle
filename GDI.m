function [gdi] = GDI(attributes)
    t = tabulate(attributes);
    p = t(:,3) / 100;
    tmp_sum = 0;
    for i = 1:length(p);
        tmp_sum = tmp_sum + p(i)^2;
    end
    gdi = 1 - tmp_sum;
    
        
