"""Lagomath CPU source package. Borrowed views carry the caller's origin."""
from vectype import Vec
from mattype import Mat, Mat22T, Mat33T, Mat44T, Mat22f, Mat33f, Mat44f
from vecviewtype import VecView
from matviewtype import MatView
from ndarraytype import NDArray

from smallmatops import Mat22Ops, Mat33Ops, Mat44Ops
from vectype import (
    Vec3Ops,
    Vec3SliceOps,
    Vec2T,
    Vec3T,
    Vec2f,
    Vec3f,
    initVec2,
    initVec3,
)

import cholesky
import gauss
import linsolve
import lu
import matops
import rowops
import sliceops
import vecops

from matviewops import MatViewOps
from ndarrayops import NDArrayOps
from ndarraytype import matchArrayDims

from mappedndarraytype import MappedNDArray
import vectype
import mattype
import vecviewtype
import matviewtype
import ndarraytype
