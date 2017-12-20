function [res] = offline_face_iter(iterations)
res = 0;
for i = 1 : iterations
  a = offline_face_recognition();
  res = res + a;
end
res = res/iterations;
end;
  