* Using general form, we have 36 variables and 36 constraints
Sets
    i   / 1*36 /
    j   / 1*36 /;
* Using instructions at https://www.gams.com/mccarl/gdxusage.pdf
* Load parameters from GDX file
$gdxin P2_Gen
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
cond(j)..   sum(i, A(j, i) * x(i)) =l= b(j);
pos(i)..    x(i) =g= 0;

Model Problem2 / all /;
solve Problem2 using LP minimizing cost;