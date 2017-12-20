function [res] = hard_vs_soft_hfs_iter(iterations)
res= [0; 0]
for i = 1 : iterations
  a = hard_vs_soft_hfs();
  res = res + a;
end
res = res/iterations;
end;
  