import random
from tqdm import trange
from tinygrad.helpers import getenv, DEBUG, colored
from tinygrad.shape.shapetracker import ShapeTracker
from test.external.fuzz_shapetracker import shapetracker_ops
from test.external.fuzz_shapetracker import do_permute, do_reshape_split_one, do_reshape_combine_two, do_flip, do_pad
from test.unit.test_shapetracker_math import st_equal, MultiShapeTracker
from tinygrad.shape.view import _merge_dims, strides_for_shape

def get_real_view(shape, strides, offset, mask):
  real_shape = tuple(y-x for x,y in mask) if mask else shape
  offset = offset + sum(st * (s-1) for s,st in zip(real_shape, strides) if st<0)
  real_offset = offset + (sum(x*st for (x,_),st in zip(mask, strides)) if mask else 0)
  real_real_shape = [s for s,st in zip(real_shape, strides) if st]
  strides = [abs(st) if isinstance(st,int) else st for st in strides if st]
  return real_real_shape, strides, real_offset

def fuzz_plus():
  m = MultiShapeTracker([ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))])
  for _ in range(4): random.choice(shapetracker_ops)(m)
  backup = m.sts[0]
  m.sts.append(ShapeTracker.from_shape(m.sts[0].shape))
  for _ in range(4): random.choice(shapetracker_ops)(m)
  vm2 = backup.views[-1]
  vm1 = m.sts[1].views[-1]
  print(f'vm2 ----> {vm2}')
  print(f'vm1 ----> {vm1}')
  p = _merge_dims(vm2.shape, vm2.strides)
  e = _merge_dims(vm1.shape, vm1.strides)
  print(p)
  print(e)
  s2, st2, o2 = get_real_view(vm2.shape, vm2.strides, vm2.offset, vm2.mask)
  s1, st1, o1 = get_real_view(vm1.shape, vm1.strides, vm1.offset, vm1.mask)
  print(s2, st2, o2)
  print(s1, st1, o1)
  if vm2.mask is not None:
    m2 = tuple((x,s-y) if st != 0 else (0,0) for (x,y),s,st in zip(vm2.mask, vm2.shape, vm2.strides))
    print(f'm2 {m2}')
  if vm1.mask is not None: 
    m1 = tuple((x,s-y) if st != 0 else (0,0) for (x,y),s,st in zip(vm1.mask, vm1.shape, vm1.strides))
    print(f'm1 {m1}')
  st_sum = backup + m.sts[1]
  return m.sts[0], st_sum

# shrink and expand aren't invertible, and stride is only invertible in the flip case
invertible_shapetracker_ops = [do_permute, do_reshape_split_one, do_reshape_combine_two, do_flip, do_pad]

def fuzz_invert():
  start = ShapeTracker.from_shape((random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))
  m = MultiShapeTracker([start])
  for _ in range(8): random.choice(invertible_shapetracker_ops)(m)
  inv = m.sts[0].invert(start.shape)
  st_sum = (m.sts[0] + inv) if inv else None
  return start, st_sum

if __name__ == "__main__":
  random.seed(42)
  total = getenv("CNT", 1000)
  for fuzz in [globals()[f'fuzz_{x}'] for x in getenv("FUZZ", "invert,plus").split(",")]:
    for _ in trange(total, desc=f"{fuzz}"):
      st1, st2 = fuzz()
      eq = st_equal(st1, st2)
      if DEBUG >= 1:
        print(f"EXP: {st1}")
        print(f"GOT: {st2}")
        print(colored("****", "green" if eq else "red"))
      #if not eq: exit(0)
