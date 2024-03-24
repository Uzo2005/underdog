import streams, json, strutils, sequtils

const bufferSize = 12 * 1024 * 1024 #12 megabytes
let
    weightsStream = newFilestream("./stablelm_1_6b_model/stable_lm_1_6b_8bdf317e2b35ab5c8009cbb6c7ce495e4e608a6b9b843d44054edf25b8c5860d.safetensors", fmRead)

var myBuf: array[bufferSize, char]
if not(isNil(weightsStream)):
    var header: array[8, byte]
    weightsStream.peek(header)
    let headerLen = (cast[ptr uint64](addr (header)))[]
    echo "Header has length: ", headerLen
    weightsStream.setPosition(8)
    echo "Reading in ", weightsStream.readData(addr myBuf, headerLen.int), " bytes into the buffer"

    let metadata = parseJson(myBuf[0..<headerLen].join)
    # echo myBuf[0..<headerLen].join
    # weightsStream.setPosition(3289030656)
    # echo "Reading in ", weightsStream.readData(addr myBuf, 38864), " bytes into the buffer"
    # echo weightsStream.getPosition()

    # echo metadata["model.layers.14.input_layernorm.weight"]["shape"]


    # weight

    # while weightsStream.readline(line):
    #     echo line
    weightsStream.close()


