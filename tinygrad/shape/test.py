#!/usr/bin/env python3

from tinygrad.shape.shapetracker import ShapeTracker, merge_views, expr_node_mask, expr_node
from tinygrad.shape.view import _merge_dims, strides_for_shape, View
from tinygrad.shape.symbolic import Variable, sym_infer
from tinygrad.helpers import prod

#a = ShapeTracker.from_shape((2, 5))
a = View.create(shape=(20, 1), strides=(1, 0), offset=0, mask=((0, 10), (0, 1)))
b = View.create(shape=(3, 10, 1), strides=(0, -2, 0), offset=19, mask=((2, 3), (0, 10), (0, 1)))
c = ShapeTracker((a,b))
#c = ShapeTracker((b,))
#print(c)
#print(c.shape)
idx = Variable("idx", 0, prod(c.shape)-1)
#print(idx)
'''
'''
#valid = expr_node_mask(b, idx)
#print(valid)
c_idx, c_valid = c.expr_node(idx)
#print(c_idx)
#print(c_valid)
#print(c_idx)
#print(c_valid)
for i in range(idx.min, idx.max):
    c_off = sym_infer(c_idx, {idx: i})
    #print(c_off)
    c_v = sym_infer(c_valid, {idx: i})
    print(c_v)

'''
n = expr_node(c.views[-1], idx)
print(n)
m = expr_node_mask(c.views[-1], idx)
print(m)'''
'''
valid = expr_node_mask(c.views[-2], n, m)
print(valid)
idx1 = expr_node(c.views[-2], n)
print(idx1)
'''
'''
d = View.create(shape=(3, 10, 1), strides=(0, -2, 0), offset=19, mask=((2, 3), (5, 10), (0, 1)))
d = ShapeTracker((d,))
#print(d)
#print(d.shape)
idx2 = Variable("idx", 0, prod(c.shape)-1)
#print(idx)
d_idx, d_valid = d.expr_node(idx)
print(d_idx)
print(d_valid)
for i in range(idx2.min, idx2.max):
    d_off = sym_infer(d_idx, {idx2: i})
    #print(d_off)
    d_v = sym_infer(d_valid, {idx2: i})
    #print(d_v)
'''
'''
a = View.create(shape=(5, 1, 8, 6), strides=(0, 0, 6, 1), offset=0, mask=((2, 4), (0, 1), (0, 7), (0, 6)))
b = View.create(shape=(5, 1, 1, 8, 3, 1), strides=(48, 0, 0, 6, -2, 0), offset=5, mask=None)
c = ShapeTracker((a,b))


d = _merge_dims(shape=(5, 1, 8, 6), strides=(0, 0, 6, 1))
e = _merge_dims(shape=(5, 1, 1, 8, 3, 1), strides=(48, 0, 0, 6, -2, 0))
print(d)
print(e)

print(strides_for_shape((5,1,1,8,3,1)))
'''