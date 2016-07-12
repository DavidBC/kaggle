function [gdi_X] = GDI_X(attributes, variables)
    t = tabulate(variables);
    %[n, ~] = size(t);
    gdi_X = 0;
    for i = 1:2;
        gdi = GDI(attributes(variables == t(i, 1)));
        p = t(i, 3) / 100;
        gdi_X = gdi_X + (p*gdi);
    end

    