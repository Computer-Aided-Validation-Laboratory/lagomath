"""Independent references for the retained method and ownership surface."""
import numpy as np
import scipy.linalg


def generate(add):
    for dtype in ['f32','f64']:
        dt=np.float32 if dtype=='f32' else np.float64
        def case(op,rep,a,expected,b=(),shape=None,index=(),alpha=2,label='methods',source='numpy',error=None):
            add('extended',op,dtype,dict(a=np.asarray(a).tolist(),b=np.asarray(b).tolist(),x=[],shape=shape or [len(a)],index=list(index),alpha=alpha),expected,error=error,representation=rep,label=label,source=source)
        a=np.array([1,3,6,7,8,1,-2,-3,0,5],dtype=dt)
        for rep in ['Vec','VecView','slice']:
            case('extremaIndices',rep,a,[8,4,-3,7])
            case('scale',rep,a,a*2)
            case('legacyNorms',rep,a,[np.dot(a,a),np.dot(a,a),np.linalg.norm(a)])
            for op,fn,values in [('applySqrt',np.sqrt,np.ones(7,dtype=dt)),('applyAtan',np.arctan,np.zeros(7,dtype=dt)),('applyExp',np.exp,np.zeros(7,dtype=dt))]:
                case(op,rep,values,fn(values))
        case('coordinates','Vec',[2,2,7,2],[2,2,7,2])
        for rep in ['Vec','VecView']:
            for op,expected in [('zeros',[0]*3),('ones',[1]*3),('fill',[2]*3)]:case(op,rep,[9]*3,expected)
        case('initVec2','Vec',[2,3],[2,3]);case('initVec3','Vec',[1,2,3],[1,2,3])
        # Preserve the existing Zig middle-component convention during this migration.
        for a,b,label in [([1,0,0],[0,1,0],'basis'),([1,2,3],[4,5,6],'signed')]:
            expected=np.cross(a,b);expected[1]*=-1
            case('cross','Vec',a,expected,b=b,label=label)
        vals=[2,4,6,-1,5,7,3,-2,8]
        case('componentExtrema','Vec',vals,[3,5,8,-1,-2,6],shape=[3,3])
        for rep in ['Vec','VecView','slice']:
            case('extremaIndices',rep,[3,3,-2,-2],[3,0,-2,2],label='ties')
        for rep in ['slice']:
            for op in ['addChecked','subChecked','mulChecked','divChecked','mulScalChecked']:
                case(op,rep,[1,2,3],None,b=[1,2],error='LengthMismatch')
            case('partialApply',rep,[0,0,99,99],[1,1,99,99],b=[0,0])
            case('rangeLen',rep,[0,1,.3],[4])
        for r,c in [(2,2),(3,3),(4,4),(2,3),(3,2)]:
            a=np.arange(1,r*c+1,dtype=dt).reshape(r,c);b=a+4
            flat=a.ravel().tolist();shape=[r,c,c]
            for rep in ['Mat','MatView']:
                for op,expected in [('zeros',np.zeros_like(a)),('ones',np.ones_like(a)),('fill',np.full_like(a,2)),('identity',np.eye(r,c)),('diag',np.eye(r,c)*2),('scale',a*2),('add',a+b),('sub',a-b),('row',a[-1]),('column',a[:,0])]:
                    if rep=='MatView' and op=='ones':continue
                    case(op,rep,flat,expected.ravel(),b=b.ravel(),shape=shape,index=[r-1,0],label=f'{r}x{c}')
                for row_b,label in [(r-1,'firstLast'),(0,'sameRow')]:
                    for scale in [0,-.5]:
                        for op in ['swapRowsMethod','scaleRowMethod','addScaledRowMethod']:
                            expected=a.copy()
                            if op=='swapRowsMethod':expected[[0,row_b]]=expected[[row_b,0]]
                            elif op=='scaleRowMethod':expected[0]*=scale
                            else:expected[row_b]+=expected[0]*scale
                            if rep=='Mat':case(op,rep,flat,expected.ravel(),shape=shape,index=[0,row_b],alpha=scale,label=f'{r}x{c}.{label}.{scale}')
                if rep=='MatView':
                    for op,expected in [('mulInPlace',a*b),('divInPlace',a/b),('mulOut',a*b),('divOut',a/b)]:case(op,rep,flat,expected.ravel(),b=b.ravel(),shape=shape,label=f'{r}x{c}')
                    case('columnExtrema',rep,flat,[a[:,0].min(),a[:,0].max()],shape=shape,index=[0],label=f'{r}x{c}')
                    expected=a.copy();expected[-1,-1]=2
                    case('flatAccess',rep,flat,expected.ravel(),shape=shape,index=[r-1,c-1],label=f'{r}x{c}')
                    case('flatHelpers',rep,flat,[r*c-1,(r-1)*c,a[-1,-1]],shape=shape,index=[r-1,c-1],label=f'{r}x{c}')
                    if r==c:case('transposeSquare',rep,flat,a.T.ravel(),shape=shape,label=f'{r}x{c}')
                else:
                    case('initRows',rep,flat,flat,shape=shape,label=f'{r}x{c}')
                    expected=a.copy();expected[-1,:]=2
                    case('insertRow',rep,flat,expected.ravel(),shape=shape,index=[r-1,0],label=f'{r}x{c}')
                    expected=a.copy();expected[:,0]=2
                    case('insertColumn',rep,flat,expected.ravel(),shape=shape,index=[0,0],label=f'{r}x{c}')
            if r>=2 and c>=2:
                case('submatrix','Mat',flat,a[:2,:2].ravel(),shape=shape,index=[0,0],label=f'{r}x{c}')
                expected=a.copy();expected[:2,:2]=2
                case('insertSubmatrix','Mat',flat,expected.ravel(),shape=shape,index=[0,0],label=f'{r}x{c}')
        case('adj','Mat',[1,2,3,4],[4,-2,-3,1],shape=[2,2,2])
        case('mulVec3','Mat',list(range(1,17)),np.array(list(range(1,17))).reshape(4,4)[:3,:3]@np.array([1,2,3])+np.array([4,8,12]),b=[1,2,3],shape=[4,4,4])
        for n in [2,3,4]:
            a=np.arange(1,n*n+1,dtype=dt).reshape(n,n)
            for op,expected in [('mulMatMethod',a@a),('mulMatRectMethod',a@a),('mulSquareView',a@a),('mulView',a@a),('scaleOut',a*2),('addOut',a+a),('subOut',a-a)]:
                rep='Mat' if 'Method' in op else 'MatView'
                case(op,rep,a.ravel(),expected.ravel(),b=a.ravel(),shape=[n,n,n],label=str(n))
            case('mulVecMethod','Mat',a.ravel(),a@np.ones(n,dtype=dt),b=np.ones(n,dtype=dt),shape=[n,n,n],label=str(n))
            case('mulVecView','MatView',a.ravel(),a@np.ones(n,dtype=dt),b=np.ones(n,dtype=dt),shape=[n,n,n],label=str(n))
        for rep in ['Vec','VecView']:
            case('permutation',rep,[1,2,3],[3,1,2],shape=[3],index=[2,0,1])
        expected=np.arange(1,17,dtype=dt).reshape(4,4);expected[2:4,2:4]=2
        case('insertMat22','Mat',list(range(1,17)),expected.ravel(),shape=[4,4,4],index=[2,2])
        case('invThreshold','Mat',[1,0,0,1e-14],None,shape=[2,2,2],alpha=1e-12,error='SingularMatrix')
        for label,perm in [('duplicate',[0,0,2]),('outOfBounds',[0,1,3])]:
            case('badPermutation','MatView',np.eye(3).ravel(),None,b=[1,2,3],shape=[3,3],index=perm,label=label,error='InvalidPermutation')
        for op in ['gaussScratch','luScratch','choleskyScratch']:
            case(op,'MatView',np.eye(3).ravel(),None,b=[1,2,3],shape=[3,3],error='ShapeMismatch')
        case('luFactoredZero','MatView',[0,0,0,1],None,b=[1,1],shape=[2,2],index=[0,1],error='SingularMatrix')
        case('choleskyFactoredZero','MatView',[0,0,0,1],None,b=[1,1],shape=[2,2],error='NotPositiveDefinite')
        # Buffer sentinels and inherited oversized backing-slice semantics.
        case('rowMutation','MatView',[99,1,2,3,4,5,6,99],[99,1,2,3,2,2,2,99],shape=[2,3,3],index=[1])
        for shape in [[1],[17],[2,3],[3,2],[16,17],[64,64],[2,3,4],[1,4,7],[8,9,10],[2,3,4,5],[1,2,1,7],[4,5,6,7],[2,2,2,2,2],[2,1,2,1,2,1]]:
            n=int(np.prod(shape));data=np.arange(n,dtype=dt);idx=[d-1 for d in shape]
            for indices,label in [([0]*len(shape),'first'),([d//2 for d in shape],'middle'),(idx,'last')]+[([shape[k]-1 if k==axis else 0 for k in range(len(shape))],'axis'+str(axis)) for axis in range(len(shape))]:
                case('getset','NDArray',data,np.concatenate([data[:int(np.ravel_multi_index(indices,shape))],[2],data[int(np.ravel_multi_index(indices,shape))+1:]]),shape=shape,index=indices,label='x'.join(map(str,shape))+'.'+label)
            if n<=120:
                for op,expected in [('fill',np.full(n,2)),('scale',data*2),('add',data+2),('sub',data-2),('mulInPlace',data*2),('divInPlace',data/2),('addOut',data+2),('subOut',data-2),('mulOut',data*2),('divOut',data/2),('dupe',np.r_[data,data]),('emptyPrefix',data),('initFlat',np.full(n,2))]:case(op,'NDArray',data,expected,shape=shape,b=[2]*n,label='x'.join(map(str,shape)))
        case('planeTail','NDArray',list(range(27)),list(range(12,27)),shape=[2,3,4],index=[1])
        case('getSlice','NDArray',list(range(24)),list(range(4,8)),shape=[2,3,4],index=[0,1,0],alpha=1)
        case('matchDims','NDArray',list(range(6)),[1,0],shape=[2,3])
        case('extractMat','NDArray',list(range(24)),np.arange(24).reshape(2,3,4)[1,:,:].ravel(),shape=[2,3,4],index=[1,0,0])
        for n in [2,3,6,12]:
            rng=np.random.default_rng(54321+n);j=rng.normal(size=(n+3,n));a=np.asarray(j.T@j+np.eye(n),dtype=dt)
            rhs=np.stack([np.arange(n,dtype=dt)+k for k in [1,3,-2]])
            for rep in ['Mat','MatView']:
                for op in ['luReconstruction','choleskyReconstruction']:
                    case(op,rep,a.ravel(),a.ravel(),shape=[n,n],source='mathematical_identity',label=str(n))
                for op in ['luMultiple','choleskyMultiple']:
                    expected=np.stack([scipy.linalg.solve(a.astype(float),b.astype(float),assume_a='pos') for b in rhs])
                    case(op,rep,a.ravel(),expected.ravel(),b=rhs.ravel(),shape=[n,n],source='scipy',label=str(n))
    add('extended','integerArithmetic','i32',dict(a=[2,2,2,2],b=[],x=[],shape=[4],index=[2],alpha=7),[2,2,7,2,13],representation='Vec',label='historical',source='hand')
