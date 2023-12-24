#!/usr/bin/env python3

from tinygrad.shape.shapetracker import ShapeTracker, merge_views
from tinygrad.shape.view import _merge_dims, strides_for_shape, View

a = ShapeTracker.from_shape((2, 5))
a = a.pad( ((2,0), (0,0)) )
print(a)



