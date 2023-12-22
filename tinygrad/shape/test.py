from tinygrad.shape.shapetracker import ShapeTracker

a = ShapeTracker.from_shape((2, 5))
x = a.pad( ((2,0), (0,0)) )
x = x.reshape( (2,2,5) )
x1 = x.reshape( (4,5) )
x1 = x1.reshape( (2,2,5) )

print(x)
print(x1.simplify())


