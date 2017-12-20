function obj = func_obj(f,C,y)
obj = (f-y)'*C*(f-y) + f'*L*f;
end