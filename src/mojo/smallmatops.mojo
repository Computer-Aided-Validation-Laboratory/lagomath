from mattype import Mat
from vectype import Vec
from std.math import abs, isfinite
from std.collections import Array


struct Mat22Ops:
    @staticmethod
    def adj[dtype: DType](mat22: Mat[2, 2, dtype]) -> Mat[2, 2, dtype]:
        var adjoint = Mat[2, 2, dtype]()
        adjoint.set(0, 0, mat22.get(1, 1))
        adjoint.set(1, 1, mat22.get(0, 0))
        adjoint.set(0, 1, -1 * mat22.get(0, 1))
        adjoint.set(1, 0, -1 * mat22.get(1, 0))
        return adjoint^

    @staticmethod
    def det[dtype: DType](mat22: Mat[2, 2, dtype]) -> Scalar[dtype]:
        return (mat22.get(0, 0) * mat22.get(1, 1)) - (
            mat22.get(0, 1) * mat22.get(1, 0)
        )

    @staticmethod
    def inv[dtype: DType](mat22: Mat[2, 2, dtype]) -> Mat[2, 2, dtype]:
        var inv_mat = Mat22Ops.adj(mat22)
        var mat_det: Scalar[dtype] = Mat22Ops.det(mat22)
        debug_assert(isfinite(mat_det) and mat_det != 0, "invertible matrix")
        inv_mat = inv_mat.mulScal(1 / mat_det)
        return inv_mat^

    @staticmethod
    def invChecked[
        dtype: DType
    ](mat22: Mat[2, 2, dtype], min_abs_det: Scalar[dtype]) raises -> Mat[
        2, 2, dtype
    ]:
        var mat_det = Mat22Ops.det(mat22)
        if not isfinite(mat_det) or abs(mat_det) <= min_abs_det:
            raise Error("SingularMatrix")
        return Mat22Ops.inv(mat22)


struct Mat33Ops:
    @staticmethod
    def det[dtype: DType](mat33: Mat[3, 3, dtype]) -> Scalar[dtype]:
        var sub_dets = Array[Scalar[dtype], 3](fill=0)
        sub_dets[0] = mat33.get(0, 0) * (
            (mat33.get(1, 1) * mat33.get(2, 2))
            - (mat33.get(1, 2) * mat33.get(2, 1))
        )
        sub_dets[1] = mat33.get(0, 1) * (
            (mat33.get(1, 0) * mat33.get(2, 2))
            - (mat33.get(1, 2) * mat33.get(2, 0))
        )
        sub_dets[2] = mat33.get(0, 2) * (
            (mat33.get(1, 0) * mat33.get(2, 1))
            - (mat33.get(1, 1) * mat33.get(2, 0))
        )
        return sub_dets[0] - sub_dets[1] + sub_dets[2]

    @staticmethod
    def inv[dtype: DType](mat33: Mat[3, 3, dtype]) -> Mat[3, 3, dtype]:
        var inv33 = Mat[3, 3, dtype]()
        var mat_det = Mat33Ops.det(mat33)
        debug_assert(isfinite(mat_det) and mat_det != 0, "invertible matrix")
        var detm = 1 / mat_det
        inv33.mat[0] = detm * (
            mat33.get(1, 1) * mat33.get(2, 2)
            - mat33.get(1, 2) * mat33.get(2, 1)
        )
        inv33.mat[1] = -detm * (
            mat33.get(0, 1) * mat33.get(2, 2)
            - mat33.get(0, 2) * mat33.get(2, 1)
        )
        inv33.mat[2] = detm * (
            mat33.get(0, 1) * mat33.get(1, 2)
            - mat33.get(0, 2) * mat33.get(1, 1)
        )
        inv33.mat[3] = -detm * (
            mat33.get(1, 0) * mat33.get(2, 2)
            - mat33.get(1, 2) * mat33.get(2, 0)
        )
        inv33.mat[4] = detm * (
            mat33.get(0, 0) * mat33.get(2, 2)
            - mat33.get(0, 2) * mat33.get(2, 0)
        )
        inv33.mat[5] = -detm * (
            mat33.get(0, 0) * mat33.get(1, 2)
            - mat33.get(0, 2) * mat33.get(1, 0)
        )
        inv33.mat[6] = detm * (
            mat33.get(1, 0) * mat33.get(2, 1)
            - mat33.get(1, 1) * mat33.get(2, 0)
        )
        inv33.mat[7] = -detm * (
            mat33.get(0, 0) * mat33.get(2, 1)
            - mat33.get(0, 1) * mat33.get(2, 0)
        )
        inv33.mat[8] = detm * (
            mat33.get(0, 0) * mat33.get(1, 1)
            - mat33.get(0, 1) * mat33.get(1, 0)
        )
        return inv33^

    @staticmethod
    def invChecked[
        dtype: DType
    ](mat33: Mat[3, 3, dtype], min_abs_det: Scalar[dtype]) raises -> Mat[
        3, 3, dtype
    ]:
        var mat_det = Mat33Ops.det(mat33)
        if not isfinite(mat_det) or abs(mat_det) <= min_abs_det:
            raise Error("SingularMatrix")
        return Mat33Ops.inv(mat33)


struct Mat44Ops:
    @staticmethod
    def det[dtype: DType](mat: Mat[4, 4, dtype]) -> Scalar[dtype]:
        var mat_a = mat.getSubMat[2, 2](0, 0)
        var mat_b = mat.getSubMat[2, 2](0, 2)
        var mat_c = mat.getSubMat[2, 2](2, 0)
        var mat_d = mat.getSubMat[2, 2](2, 2)
        var det_a = Mat22Ops.det(mat_a)
        var det_b = Mat22Ops.det(mat_b)
        var det_c = Mat22Ops.det(mat_c)
        var det_d = Mat22Ops.det(mat_d)
        var adj_a = Mat22Ops.adj(mat_a)
        var adj_d = Mat22Ops.adj(mat_d)
        var adj_ab = adj_a.mulMat(mat_b)
        var adj_dc = adj_d.mulMat(mat_c)
        var adj_ab_dc = adj_ab.mulMat(adj_dc)
        return det_a * det_d + det_b * det_c - adj_ab_dc.trace()

    @staticmethod
    def inv[dtype: DType](mat: Mat[4, 4, dtype]) -> Mat[4, 4, dtype]:
        var mat_a = mat.getSubMat[2, 2](0, 0)
        var mat_b = mat.getSubMat[2, 2](0, 2)
        var mat_c = mat.getSubMat[2, 2](2, 0)
        var mat_d = mat.getSubMat[2, 2](2, 2)
        var det_a = Mat22Ops.det(mat_a)
        var det_b = Mat22Ops.det(mat_b)
        var det_c = Mat22Ops.det(mat_c)
        var det_d = Mat22Ops.det(mat_d)
        var adj_a = Mat22Ops.adj(mat_a)
        var adj_d = Mat22Ops.adj(mat_d)
        var adj_ab = adj_a.mulMat(mat_b)
        var adj_dc = adj_d.mulMat(mat_c)
        var adj_ab_dc = adj_ab.mulMat(adj_dc)
        var det_m: Scalar[dtype] = (
            det_a * det_d + det_b * det_c - adj_ab_dc.trace()
        )
        debug_assert(isfinite(det_m) and det_m != 0, "invertible matrix")
        var inv_a = mat_a.mulScal(det_d)
        var b_adj_dc = mat_b.mulMat(adj_dc)
        inv_a = inv_a.sub(b_adj_dc)
        inv_a = Mat22Ops.adj(inv_a)
        var inv_b = mat_c.mulScal(det_b)
        var adj_ab_adj = Mat22Ops.adj(adj_ab)
        var d_adj_ab_adj = mat_d.mulMat(adj_ab_adj)
        inv_b = inv_b.sub(d_adj_ab_adj)
        inv_b = Mat22Ops.adj(inv_b)
        var inv_c = mat_b.mulScal(det_c)
        var adj_dc_adj = Mat22Ops.adj(adj_dc)
        var a_adj_dc_adj = mat_a.mulMat(adj_dc_adj)
        inv_c = inv_c.sub(a_adj_dc_adj)
        inv_c = Mat22Ops.adj(inv_c)
        var inv_d = mat_d.mulScal(det_a)
        var c_adj_ab = mat_c.mulMat(adj_ab)
        inv_d = inv_d.sub(c_adj_ab)
        inv_d = Mat22Ops.adj(inv_d)
        var mat_inv = Mat[4, 4, dtype].initIdentity()
        mat_inv.insertSubMat(0, 0, inv_a)
        mat_inv.insertSubMat(0, 2, inv_b)
        mat_inv.insertSubMat(2, 0, inv_c)
        mat_inv.insertSubMat(2, 2, inv_d)
        mat_inv = mat_inv.mulScal(1 / det_m)
        return mat_inv^

    @staticmethod
    def invChecked[
        dtype: DType
    ](mat: Mat[4, 4, dtype], min_abs_det: Scalar[dtype]) raises -> Mat[
        4, 4, dtype
    ]:
        var mat_det = Mat44Ops.det(mat)
        if not isfinite(mat_det) or abs(mat_det) <= min_abs_det:
            raise Error("SingularMatrix")
        return Mat44Ops.inv(mat)

    @staticmethod
    def insertMat22[
        dtype: DType
    ](
        mut matrix: Mat[4, 4, dtype],
        block: Mat[2, 2, dtype],
        row: Int,
        col: Int,
    ):
        matrix.insertSubMat(row, col, block)

    @staticmethod
    def mulVec3[
        dtype: DType
    ](matrix: Mat[4, 4, dtype], vector: Vec[3, dtype]) -> Vec[3, dtype]:
        var result = Vec[3, dtype]()
        for r in range(3):
            var value: Scalar[dtype] = 0
            for c in range(3):
                value += matrix.get(r, c) * vector.get(c)
            value += matrix.get(r, 3)
            result.set(r, value)
        return result^
