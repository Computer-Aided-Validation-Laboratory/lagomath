@fieldwise_init
struct ValIdx[dtype: DType](Copyable, Movable):
    var val: Scalar[Self.dtype]
    var idx: Int
