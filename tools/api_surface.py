"""Explicit stable method contracts, including constructors and conveniences."""
SURFACES={
'Vec':('vectype',{'initFill':['fill'],'initOnes':['ones'],'initZeros':['zeros'],'initSlice':['valueCopy'],'get':['valueCopy'],'set':['valueCopy'],'x':['coordinates'],'y':['coordinates'],'z':['coordinates'],'w':['coordinates'],'add':['add'],'sub':['sub'],'mulScal':['scale'],'dot':['dot'],'normL1':['normL1'],'normL2':['normL2'],'normL2Square':['normL2Square'],'norm':['legacyNorms'],'squaredNorm':['legacyNorms'],'vecLen':['legacyNorms'],'max':['extremaIndices'],'min':['extremaIndices'],'sum':['sum'],'mean':['mean'],'apply':['applySqrt','applyAtan','applyExp'],'vecPrint':[]}),
'VecView':('vecviewtype',{'init':['viewMutation'],'fill':['fill'],'get':['viewMutation'],'set':['viewMutation'],'addInPlace':['add'],'subInPlace':['sub'],'mulInPlace':['mul'],'divInPlace':['div'],'mulScalInPlace':['scale'],'applyInPlace':['applySqrt','applyAtan','applyExp'],'dot':['dot'],'normL1':['normL1'],'normL2':['normL2'],'normL2Square':['normL2Square'],'norm':['legacyNorms'],'squaredNorm':['legacyNorms'],'vecLen':['legacyNorms'],'max':['extremaIndices'],'min':['extremaIndices'],'sum':['sum'],'mean':['mean'],'vecPrint':[]}),
'Mat':('mattype',{'initFill':['fill'],'initZeros':['zeros'],'initOnes':['ones'],'initDiag':['diag'],'initIdentity':['identity'],'initSlice':['valueCopy'],'initRows':['initRows'],'get':['valueCopy'],'set':['valueCopy'],'getRowVec':['row'],'getColVec':['column'],'getSubMat':['submatrix'],'insertRowVec':['insertRow'],'insertColVec':['insertColumn'],'insertSubMat':['insertSubmatrix'],'transpose':['transpose'],'trace':['trace'],'add':['add'],'sub':['sub'],'mulScal':['scale'],'mulVec':['mulVecMethod'],'mulMat':['mulMatMethod'],'mulMatRect':['mulMatRectMethod'],'matPrint':[]}),
'MatView':('matviewtype',{'init':['matmul'],'initAlloc':['transposeSquare'],'fill':['fill'],'fillDiag':['diag'],'identity':['identity'],'get':['matmul'],'set':['matmul'],'flatIndex':['flatAccess'],'rowBase':['flatHelpers'],'getFlat':['flatHelpers'],'setFlat':['flatAccess'],'transposeSquare':['transposeSquare'],'transpose':['transpose'],'trace':['trace'],'addInPlace':['add'],'subInPlace':['sub'],'mulInPlace':['mulInPlace'],'divInPlace':['divInPlace'],'mulScalInPlace':['scale'],'getSlice':['rowMutation'],'matPrint':[],'minByRow':['columnExtrema'],'maxByRow':['columnExtrema']}),
'MatViewOps':('matviewops',{'add':['addOut'],'sub':['subOut'],'mulElemWise':['mulOut'],'divElemWise':['divOut'],'mulScal':['scaleOut'],'mulVec':['mulVecView'],'mulSquare':['mulSquareView'],'mul':['mulView']}),
'NDArray':('ndarraytype',{'init':['getFlatIdx'],'initFlat':['initFlat'],'deinit':[],'set':['getset'],'get':['getset'],'getFlat':['dupe'],'setFlat':['getset'],'getFlatIdx':['getFlatIdx'],'planeBase':['planeBase'],'subBase2':['subBase2'],'offset2':['offset2'],'offset3':['offset3'],'offset4':['offset4'],'calcFlatStride':['calcFlatStride'],'fill':['fill'],'addInPlace':['add'],'subInPlace':['sub'],'mulInPlace':['mulInPlace'],'divInPlace':['divInPlace'],'mulScalInPlace':['scale'],'getSlice':['getSlice'],'getPlaneSlice':['planeTail'],'fixedPrefixView':['fixedPrefixView','emptyPrefix'],'dupe':['dupe']}),
'NDArrayOps':('ndarrayops',{'add':['addOut'],'sub':['subOut'],'mulElemWise':['mulOut'],'divElemWise':['divOut'],'extractMat':['extractMat']}),
'Vec3Ops':('vectype',{'cross':['cross']}),'Vec3SliceOps':('vectype',{'min':['componentExtrema'],'max':['componentExtrema']}),
'Mat22Ops':('smallmatops',{'adj':['adj'],'det':['det'],'inv':['inv'],'invChecked':['inv']}),
'Mat33Ops':('smallmatops',{'det':['det'],'inv':['inv'],'invChecked':['inv']}),
'Mat44Ops':('smallmatops',{'det':['det'],'inv':['inv'],'invChecked':['inv'],'insertMat22':['insertMat22'],'mulVec3':['mulVec3']}),
'MappedNDArray':('mappedndarraytype',{'deinit':[]}),
'sliceops':('sliceops',{name:[op] for name,op in [('max','extremaIndices'),('min','extremaIndices'),('sum','sum'),('mean','mean'),('exp','applyExp'),('apply','partialApply'),('rangeLen','rangeLen'),('dot','dot'),('normL1','normL1'),('normL2','normL2'),('normL2Square','normL2Square'),('norm','legacyNorms'),('vecLen','legacyNorms'),('add','add'),('sub','sub'),('mul','mul'),('div','div'),('mulScal','scale'),('addChecked','addChecked'),('subChecked','subChecked'),('mulChecked','mulChecked'),('divChecked','divChecked'),('mulScalChecked','mulScalChecked')]}),
}


def method_catalog(root,cases,benches):
    entries=[]
    SURFACES['sliceops'][1]['slicePrint']=[]
    for owner,(module,methods) in SURFACES.items():
        zigmodule={'matviewops':'matviewtype','ndarrayops':'ndarraytype','mappedndarraytype':'ndarraytype','smallmatops':'mattype'}.get(module,module)
        zig=(root/f'src/{zigmodule}.zig').read_text()
        mojo=(root/f'src/mojo/{module}.mojo').read_text()
        representation=('Mat' if owner.startswith('Mat') and owner not in ['MatView','MatViewOps','MappedNDArray'] else 'Vec' if owner.startswith('Vec3') else 'NDArray' if owner in ['NDArrayOps','MappedNDArray'] else owner)
        if owner=='MatViewOps':representation='MatView'
        if owner=='sliceops':representation='slice'
        for method,ops in methods.items():
            ids=[c['id'] for c in cases if c['representation']==representation and c['operation'] in ops and (not owner.startswith('Mat22') or c['inputs']['shape'][:2]==[2,2]) and (not owner.startswith('Mat33') or c['inputs']['shape'][:2]==[3,3]) and (not owner.startswith('Mat44') or c['inputs']['shape'][:2]==[4,4])]
            entries.append(dict(name=owner+'.'+method,zig_module=zigmodule,mojo_module=module,representation=representation,zig=bool(__import__('re').search(r'pub fn\s+'+method+r'\b',zig)),mojo=bool(__import__('re').search(r'def\s+'+method+r'\b',mojo)),kind='mechanical' if not ops else 'numerical',shared_case_ids=ids,benchmark_ids=[c['id'] for c in benches if c['representation']==representation and c['operation'] in ops],cabi=False,verification='native cleanup/type checking or printing smoke' if not ops else 'shared native numerical cases'))
        # Fail if a newly exposed method is omitted from the reviewed registry.
        if owner in ['Vec','VecView','Mat','MatView','NDArray']:
            import re
            start=re.search(r'pub fn '+owner+r'\b',zig).end()
            end=zig.find('\ntest ',start)
            segment=zig[start:end if end>=0 else len(zig)]
            boundary={'Vec':'\npub fn Vec2T','Mat':'\npub fn Mat22T','MatView':'\npub fn MatViewOps','NDArray':'\npub fn MappedNDArray'}.get(owner)
            if boundary and boundary in segment:segment=segment.split(boundary)[0]
            names=set(re.findall(r'pub fn\s+(\w+)',segment))
            missing=names-set(methods)
            if missing:raise RuntimeError(f'Unregistered public methods: {owner}: {sorted(missing)}')
    return entries
