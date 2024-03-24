#TODO can be SIMDed because every weight is a factor of 128
#Also would be very nice to have lazy evaluations of numbers

import random, math, sequtils, strformat, strutils


type Matrix* = object 
    numberOfRows*: int
    numberOfColumns*: int
    data*: ptr UncheckedArray[float32] #will load weight as f32

template `[]`*(mat: Matrix, pos: int): float32 =
    `mat`.data[pos]

template `[]`*(mat: Matrix, row, column: int): float32 = #make it easier to get value with [row, column] index
    assert column in 0..<`mat`.numberOfColumns, $column & " not within the range of 0 and " & $mat.numberOfColumns
    assert row in 0..<`mat`.numberOfRows, $row & " not within the range of 0 and " & $mat.numberOfRows
    (`mat`.data[(`row` * `mat`.numberOfColumns) + (`column`)])

template `[]=`*(mat: Matrix, row, column: int, value: float32) = #make it easier to set value at [row, column] index
    assert row in 0..<`mat`.numberOfRows, $row & " not within the range of 0 and " & $mat.numberOfRows
    assert column in 0..<`mat`.numberOfColumns, $column & " not within the range of 0 and " & $mat.numberOfColumns
    `mat`.data[(`row` * `mat`.numberOfColumns) + (`column`)] = value

template matSize*(mat: Matrix): Natural = mat.numberOfRows * mat.numberOfColumns
template sizeOfMatrixData*(mat: Matrix): int = matSize(mat) * sizeof(typeof(mat.data[0]))

template `*`*(mat: Matrix, a: float32): untyped = `a` * `mat`
template `*`*(mat: Matrix, a: int): untyped = `a`.float32 * `mat`
template `*`*(a: int, mat: Matrix): untyped = `a`.float32 * `mat`

func `$`*(mat: Matrix): string = #TODO this printing can be made prettier
    var currentRow: int
    let numberOfRows = mat.numberOfRows
    while currentRow < numberOfRows:
        for currentColumn in 0..<mat.numberOfColumns:
            result.add(fmt" {mat[currentRow, currentColumn]:f} ")
        result.add('\n')
        inc currentRow

func `$`*(mats: varargs[Matrix]): string = 
    result.add("[\n")
    for mat in mats:
        result.add($mat)
        result.add('\n')
    result.add("]\n")

proc initMat*(numberOfRows, numberOfColumns: int): Matrix =
    result.numberOfRows = numberOfRows
    result.numberOfColumns = numberOfColumns
    result.data = cast[ptr UncheckedArray[float32]](alloc0(numberOfRows * numberOfColumns * sizeof(float32)))

func shape*(mat: Matrix): string =
    result = fmt"{mat.numberOfRows}x{mat.numberOfColumns}"

proc withData*(mat: Matrix, newData: openArray[float32]): Matrix =
    assert matSize(mat) == newData.len, " The dimensions dont match"
    result = initMat(mat.numberOfRows, mat.numberOfColumns)
    for index, data in newData:
        result.data[index] = data

func populateData*(mat: var Matrix, newData: openArray[float32]) =
    assert matSize(mat) == newData.len, " The dimensions dont match"
    for index, data in newData:
        mat.data[index] = data

func reshape*(mat: var Matrix, newNumberOfRows, newNumberOfColumns: int) =
    assert newNumberOfColumns*newNumberOfRows == mat.numberOfColumns * mat.numberOfRows, "The size of the resized matrix must be same with the original matrix"
    mat.numberOfColumns = newNumberOfColumns
    mat.numberOfRows = newNumberOfRows

proc randomize*(mat: var Matrix) =
    randomize()
    for index in 0..<matSize(mat):
        mat.data[index] = rand(5.0'f32)

proc initIdentityMat*(numberOfRows: int): Matrix =
    result = initMat(numberOfRows, numberOfRows)
    for index in 0..<numberOfRows:
        result[index, index] = 1.0

func dot*(a, b: seq[float32]): float32 =
    assert a.len == b.len, "Dot Products exist for vectors of same dimensions " & $a.len & " " & $b.len
    for index in 0..<a.len:
        result += a[index] * b[index]

proc dot*(a, b: Matrix): float32 =
    assert (a.numberOfColumns == 1 and b.numberOfColumns == 1) , "Dot products only exist for single column vectors"
    assert (a.numberOfRows == b.numberOfRows) , "Dot products only exist for two vectors with same amount of rows"
    
    for index in 0..<a.numberOfRows:
        result += (a.data[index]) * (b.data[index])

proc getRow*(mat: Matrix, rowIndex: Natural): seq[float32] = #remember indexes start from 0 and not 1
    assert rowIndex in 0..<mat.numberOfRows, $rowIndex & " not within the range of 0 and " & $(mat.numberOfRows - 1)
    let startIndex = rowIndex * mat.numberOfColumns
    for column in 0..<mat.numberOfColumns:
        result.add(mat.data[startIndex + column])

func getColumn*(mat: Matrix, colIndex: Natural): seq[float32] =
    assert colIndex in 0..<mat.numberOfColumns, $colIndex & " not within the range of 0 and " & $(mat.numberOfColumns - 1)
    for row in 0..<mat.numberOfRows:
        result.add(mat.data[row*(mat.numberOfColumns) + (colIndex)])

proc `*`*(a, b: Matrix): Matrix =
    assert a.numberOfColumns == b.numberOfRows
    let commonDim = a.numberOfColumns
    result = initMat(a.numberOfRows, b.numberOfColumns)

    for y in 0..<a.numberOfRows:
        for x in 0..<b.numberOfColumns:
            for i in 0..<commonDim:
                result[y, x] = (result[y, x] + (a[y, i] * b[i, x]))


proc `+`*(a, b: Matrix): Matrix =
    assert a.numberOfColumns == b.numberOfColumns and a.numberOfRows == b.numberOfRows, "To sum two matrices, they need to be of the same dimensions"
    result = initMat(a.numberOfRows, a.numberOfColumns)

    for index in 0..<matSize(a):
        result.data[index] = a.data[index] + b.data[index]

proc `*`*(a: float32, mat: Matrix): Matrix =
    result = initMat(mat.numberOfRows, mat.numberOfColumns)

    for index in 0..<matSize(result):
        result.data[index] = a * mat.data[index]

proc softmax*(mat: Matrix) : Matrix =
    result = initMat(mat.numberOfRows, mat.numberOfColumns)
    var expSum: float32
    for index in 0..<matSize(result):
        result.data[index] = exp(mat.data[index])
        expSum += result.data[index]

    for index in 0..<matSize(result):
        result.data[index] = result.data[index] / expSum

proc scaleUp*(mat: var Matrix, scale: float32) =
    if scale != 1.0:
        mat = scale * mat

template scaleUp*(mat: var Matrix, scale: int) = scaleUp(mat, scale.float32)

template scaleDown*(mat: var Matrix, scale: int) =
    scaleUp(mat, 1 / `scale`)

template scaleDown*(mat: var Matrix, scale: float32) =
    scaleUp(mat, 1 / `scale`)

proc transposed*(mat: Matrix) : Matrix =
    result = initMat(mat.numberOfColumns, mat.numberOfRows)
    for currentRow in 0..<mat.numberOfRows:
        for currentColumn in 0..<mat.numberOfColumns:
            result[currentColumn, currentRow] = mat[currentRow, currentColumn]

proc appendColumn*(mat: var Matrix, colValues: openArray[float32]) =
    assert colValues.len == mat.numberOfRows, " The dimensions dont match "
    let 
        temp = cast[ptr UncheckedArray[float32]](alloc0(sizeof(float32) * matSize(mat)))
    
    copyMem(temp, mat.data, sizeOfMatrixData(mat)) #store our matrix data in temp
    
    var 
        tempIndex: int
        currentRow: int
        currentIndex: int

    while currentRow < (colValues.len):
        if tempIndex mod mat.numberOfColumns == 0 and tempIndex != 0:
            mat.data[currentIndex] = colValues[currentRow]
            inc currentRow
            inc currentIndex

            if currentRow == colValues.len:
                break
            
        mat.data[currentIndex] = temp[tempIndex]
        inc currentIndex
        inc tempIndex

    inc mat.numberOfColumns
    dealloc(temp) #we have no use for the temp when this function leaves the stack


func appendRow*(mat: var Matrix, rowValues: openArray[float32]) =
    assert rowValues.len == mat.numberOfColumns
    let currentSize = matSize(mat)
    for index in 0..<rowValues.len:
        mat.data[currentSize + index] = rowValues[index]
    inc mat.numberOfRows

func normalized*(values: seq[float32]): seq[float32] =
    let 
        epsilon = float32(1.001e-5)
        n = float32(values.len)
        mean = float32(values.sum / n)
        variance = values.mapit((it - mean)^2).sum / n
        standardDeviation = variance.pow(0.5).float32

    result = values.mapit((it - mean) / (standardDeviation + epsilon))

func rootMeanSquare*(nums: ptr UncheckedArray[float32], size: Natural): float32 =
    for index in 0..<size:
        result += nums[index] ^ 2
    result /= float32(size)
    result = result.pow(0.5)

template applyGammaAndBeta(mat: Matrix, gamma, beta: float32) =
    if gamma != 1.0:
        mat.scaleUp(gamma)

    if beta != 0.0:
        for index in 0..<matSize(mat):
            mat.data[index] = mat.data[index] + beta

proc batchNorm*(mat: Matrix, gamma = 1.0, beta = 0.0): Matrix = #this will be slow
    result = initMat(mat.numberOfRows, mat.numberOfColumns)

    for currentColumn in 0..<mat.numberOfColumns:
        let normalizedColumn = normalized(mat.getColumn(currentColumn))
        for currentRow in 0..<mat.numberOfRows:
            result[currentRow, currentColumn] = normalizedColumn[currentRow]

    result.applyGammaAndBeta(gamma, beta)

proc layerNorm*(mat: Matrix, gamma=1.0, beta=0.0): Matrix =
    result = initMat(mat.numberOfRows, mat.numberOfColumns)

    for currentRow in 0..<mat.numberOfRows:
        let normalizedRow = normalized(mat.getRow(currentRow))
        for currentColumn in 0..<mat.numberOfColumns:
            result[currentRow, currentColumn] = normalizedRow[currentColumn]

    result.applyGammaAndBeta(gamma, beta)

proc rmsLayerNorm*(mat: Matrix, gamma = 1.0): Matrix =
    result = initMat(mat.numberOfRows, mat.numberOfColumns)
    let rms = rootMeanSquare(mat.data, matSize(mat))

    for index in 0..<matSize(mat):
        result.data[index] = mat.data[index] / rms
    result.scaleUp(gamma)

func relu*(a: float32): float32 {.inline.} = max(0.0, a)
func sigmoid*(x: float32, b = 1.0): float32 {.inline.} = 1 / (1 + exp(- b * x))
func silu*(x: float32, b: float32): float32 {.inline.} = x * sigmoid(x, b)
template swish*(x: float32): float32 = silu(x, b = 1.0)

func swish*(mat: var Matrix) =
    for index in 0..<matSize(mat):
        mat.data[index] = swish(mat.data[index])

proc swished*(mat: Matrix): Matrix =
    result = initMat(mat.numberOfRows, mat.numberOfColumns)
    for index in 0..<matSize(mat):
        result.data[index] = swish(mat.data[index])

proc swiglu*(inputMat: Matrix, W1, V, W2: Matrix): Matrix =
    result = (swished(inputMat * W1) * (inputMat*V))*W2

proc cos*(mat: Matrix): Matrix =
    result = initMat(mat.numberOfRows, mat.numberOfColumns)

    for index in 0..<matSize(mat):
        result.data[index] = cos(mat.data[index])

proc sin*(mat: Matrix): Matrix =
    result = initMat(mat.numberOfRows, mat.numberOfColumns)

    for index in 0..<matSize(mat):
        result.data[index] = sin(mat.data[index])

proc getRopeEmbeddings*(contextLength: int, embeddingDimension: int): seq[Matrix] =
    result = newSeqWith(contextLength, initMat(embeddingDimension, embeddingDimension))
    
    let
        evenSlices = countUp(0, (embeddingDimension div 2) - 1).toSeq.mapIt(2 * it)
        oddSlices = evenSlices.mapIt(it + 1)
        positions = initMat(contextLength, 1).withData(countUp(1, contextLength).toSeq.mapIt(it.float32))
        thetaValues = initMat(1, embeddingDimension div 2).withData(countUp(0, (embeddingDimension div 2) - 1).toSeq.mapIt(float32 10000.pow(-2.0 * (it / embeddingDimension))))
        positionsWithTheta = positions * thetaValues
        cosines = cos(positionsWithTheta)
        sines = sin(positionsWithTheta)
        negativeSines = -1 * sin(positionsWithTheta)

    for position, ropeEmbedding in result:
        for index, (row, column) in evenSlices.zip(evenSlices).pairs:
            ropeEmbedding[row, column] = cosines.getRow(position)[index]

        for index, (row, column) in evenSlices.zip(oddSlices).pairs:
            ropeEmbedding[row, column] = negativeSines.getRow(position)[index]
        
        for index, (row, column) in oddSlices.zip(evenSlices).pairs:
            ropeEmbedding[row, column] = sines.getRow(position)[index]
    
        for index, (row, column) in oddSlices.zip(oddSlices).pairs:
            ropeEmbedding[row, column] = cosines.getRow(position)[index]


# #you will need a KV cache to avoid computations you have already done

when isMainModule: #test that stuff works correctly
    # assert toFloat32(toBFloat16(0b00111011000111010000000000000000'f32)) == 0.0023956298828125, " Error in converting between f32 and BF16"
    template generateData(size: int): seq[float32] = countUp(1, size).toSeq.mapIt(it.float32)

    # import times    
    # let then = cputime()

    var firstMat = initMat(3, 3)

    # echo firstMat

    # echo firstMat.withData(countUp(1, matSize(firstMat)).toSeq.mapIt(it.float32))
    
    # firstMat.populateData(countUp(1, matSize(firstMat)).toSeq.mapIt(it.float32))
    # echo firstMat

    # firstMat.randomize
    # echo firstMat

    # var identityMat = initIdentityMat(50)
    # echo identityMat

    # var 
    #     columnMat1 = initMat(5, 1).withData(generateData(5))
    #     columnMat2 = initMat(5, 1).withData(generateData(5))
    # echo columnMat1.dot(columnMat1)

    # firstMat.populateData(generateData(matSize(firstMat)))
    # # echo firstMat.getRow(0)

    # firstMat.populateData(generateData(matSize(firstMat)))
    # echo firstMat.getColumn(0)

    firstMat.populateData([1'f32, 2'f32, 1'f32, 0'f32, 1'f32, 0'f32, 2'f32, 3'f32, 4'f32])
    echo firstMat
    echo firstMat * firstMat * firstMat * firstMat * firstMat

    # var secondMat = initMat(3, 2).withData([2'f32, 5'f32, 6'f32, 7'f32, 1'f32, 8'f32])
    # echo secondMat
    # echo firstMat * secondMat
    # echo firstMat * identityMat

    # firstMat.populateData(generateData(matSize(firstMat)))
    # echo firstMat * 3

    # firstMat.populateData([1'f32, 2'f32, 1'f32, 0'f32, 1'f32, 0'f32, 2'f32, 3'f32, 4'f32])
    # echo firstMat.softmax

    # firstMat.populateData([1'f32, 2'f32, 1'f32, 0'f32, 1'f32, 0'f32, 2'f32, 3'f32, 4'f32])
    # echo firstMat
    # echo firstMat.transposed

    # firstMat.populateData(generateData(matSize(firstMat)))
    # firstMat.appendColumn([3'f32, 5'f32, 4'f32])
    # echo firstMat

    # firstMat.populateData(generateData(matSize(firstMat)))
    # firstMat.appendRow([3'f32, 5'f32, 4'f32])
    # echo firstMat

    # firstMat.populateData(generateData(matSize(firstMat)))
    # echo firstMat
    # echo firstMat.batchNorm

    # firstMat.populateData(generateData(matSize(firstMat)))
    # echo firstMat
    # echo firstMat.layerNorm

    # firstMat.populateData(generateData(matSize(firstMat)))
    # echo firstMat
    # echo firstMat.rmsLayerNorm
    
    # firstMat.populateData(generateData(matSize(firstMat)))
    # echo firstMat
    # echo firstMat.swished

    # firstMat.populateData(generateData(matSize(firstMat)))
    # echo firstMat
    # echo cos(firstMat)

    # echo getRopeEmbeddings(6, 5)

    # echo "Took ", cputime() - then, " seconds"      
