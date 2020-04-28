% Be sure to set the current directory first!

% Save problem 1 data
load('P1.mat');
iwgdx('P1.gdx', 'A', 'b', 'c');

% Save problem 2 data
load('P2.mat');
iwgdx('P2.gdx', 'A', 'b', 'c');

% Save Problem 2 data in general form
% load('P2_Gen.mat');
% iwgdx('P2_Gen.gdx', 'A', 'b', 'c');

% NOTE: we only need to convert the new A, b, c data (above), not the
% original data

% Save problem 1 original data
% load('Problem_1-3.mat');
% iwgdx('Problem_1-3.gdx', 'A', 'd_pr', 'U', 'w');

% Save problem 2 original data
% load('Problem_2-7.mat');
% iwgdx('Problem_2-7.gdx', 'A', 'b', 'epsilon', 'N');