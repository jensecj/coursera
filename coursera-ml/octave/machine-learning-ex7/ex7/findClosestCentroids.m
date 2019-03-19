function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

                                % Set K
  K = size(centroids, 1);

               % You need to return the following variables correctly.
  idx = zeros(size(X,1), 1);

        % ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

  m = size(X, 1); # number of training examples

 # naive implementation
  for i = 1:m
    x = X(i, :); # current example
    cis = zeros(K,1);

    for j = 1:K
      mu = centroids(j, :); # current centroid

      dist_sq = sum((x - mu) .^ 2);
      cis(j) = dist_sq;
    end;

    [best_ci, best_ci_index] = min(cis);
    idx(i) = best_ci_index;
  end;




##   dist = zeros(m, K);

##   for i = 1:K

##     mu = centroids(i, :); # current centroid

## # this uses automattic broadcasting to subtract mu from every row of X
##     a = (X - mu) .^ 2;
##     b = sum(a')';
##     dist(:,i) = b;
##   end;
##   ## idx
##   ## dist
##   [md, midx] = min(dist)

       % =============================================================

end
