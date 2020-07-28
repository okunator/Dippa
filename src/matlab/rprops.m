function [R] = rprops(bw, varargin)
%RPROPS Regionprops with optimized Solidity computation

% Get PROPERTIES arguments
S=find(arrayfun(@iscellstr, varargin));
K={};
if S
    K=varargin(S(1):end);
end

if find(strcmp('Solidity', K))
    K{strcmp('Solidity', K)}='ConvexHull';
    K{end+1}='Area';
end

R=regionprops(bw, K{:});
%%
if find(strcmp('Solidity',varargin(S(1):end)))
    s1=arrayfun(@(x) convex_area(x.ConvexHull(1:(end-1), :)), R);
    c=num2cell(s1);
    [R.ConvexArea]=c{:};
    c=num2cell([R.Area]./[R.ConvexArea]);
    [R.Solidity]=c{:};
end

end

function A=convex_area(hull)
    % Shoelace formula
    A1=sum(hull(:, 1).*circshift(hull(:, 2), -1));
    A2=sum(hull(:, 2).*circshift(hull(:, 1), -1));
    A=0.5*abs(A1 - A2);
end