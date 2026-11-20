from ndarraytype import NDArray


@fieldwise_init
struct MappedNDArray[dtype: DType, origin: Origin[mut=True]](Movable):
    var array: NDArray[Self.dtype, Self.origin]
    var map: List[Int]

    def deinit(mut self):
        self.array.deinit()
        self.map = List[Int]()
