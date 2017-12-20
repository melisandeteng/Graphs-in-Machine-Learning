function [mean] = hard_two_moons(iterations)
  mean = 0;
  for i = 1 : iterations
    accuracy = two_moons_hfs();
    mean = mean + accuracy;
   end
   mean = mean/iterations;
   disp(mean);
 end