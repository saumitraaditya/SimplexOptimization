* We have 1472 variables
* and 524 constraints
* This is far too many for the trial version
Sets
    i   / 1*1472 /
    j   / 1*524 /;
* Using instructions at https://www.gams.com/mccarl/gdxusage.pdf
* Load parameters from GDX file
$gdxin P1
Parameters
    A(j, i)
    b(j)
    c(i);
$load A b c
$gdxin
Variables
    x(i)
    cost;
Equations
    obj
    cond(j)
    pos(i);

* Make an alias of i so we can hold one dimension in place while iterating over the other
Alias (i, ii);

obj..       cost =e= sum(i, c(i) * x(i));
cond(j)..   sum(i, A(j, i) * x(i)) =e= b(j);
pos(i)..    x(i) =g= 0;

Model Problem1 / all /;
solve Problem1 using LP minimizing cost;
