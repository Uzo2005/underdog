#TODO stream the embeddings to save memory
import base64, strutils, sequtils, sets

const 
    tokenizer = readfile("./stablelm_1_6b_model/arcade100k_f1f50811175d9446bcd26399db7108ef2969ad94.tiktoken")
    tokenSeq = tokenizer.strip.splitLines
    allDigits = {0..9}.toSeq.mapit($it)

func encode*(input: string): seq[int] =
    let inputLen = input.len
    var currentIndex: int

    while currentIndex < inputLen:
        var longestMatchIndex: tuple[inputIndex: int, embedValue: int]
        for token in tokenSeq:
            let 
                encoding = token.strip.splitWhitespace
                bytepair = decode encoding[0]
                embedValue = parseInt encoding[1]
                bytePairLen = bytePair.len
                
            assert bytePairLen >= 0
            if (currentIndex + bytePairLen) > inputLen:
                continue
            else:
                let currentChar = $input[currentIndex]
                if (currentChar in allDigits) and (currentChar == bytePair): #current stuff is a digit and will be tokenized seperately
                    longestMatchIndex.embedValue = embedValue
                    longestMatchIndex.inputIndex = currentIndex
                    break #continue to the next token in input

                let greediestMatchIndex =  currentIndex + bytePairLen - 1
                if input[currentIndex .. greediestMatchIndex] == bytePair: #match found but we arent sure if it is the greediest match yet
                    if longestMatchIndex.inputIndex <= max(longestMatchIndex.inputIndex, greediestMatchIndex): #we have a greedier match
                        longestMatchIndex.inputIndex = greediestMatchIndex
                        longestMatchIndex.embedValue = embedValue
                
        currentIndex = longestMatchIndex.inputIndex + 1
        result.add(longestMatchIndex.embedValue)


func decode*(embedding: openArray[int]): string =
    var 
        temp = newSeq[string](embedding.len)        
        embedsDecoded: int
    
    for token in tokenSeq:
        var 
            visitedUniqueIndexes: HashSet[int] #this wil store the index of the first occurence of every (non-)duplicate embed in embedding 
        let 
            encoding = token.strip.splitWhitespace
            bytepair = decode encoding[0]
            embedValue = parseInt encoding[1]
            embeddingFirstUniqueIndex = embedding.find(embedValue)

        if  embeddingFirstUniqueIndex > -1 and (embeddingFirstUniqueIndex notin visitedUniqueIndexes): #okay a new match has been found
            visitedUniqueIndexes.incl(embeddingFirstUniqueIndex)
            let numberOfOccurences = embedding.count(embedValue)
            var numberOfOccurencesMet: int
            for index, embed in embedding:
                if embed == embedValue:
                    inc numberOfOccurencesMet
                    # debugEcho "decoding ", embedValue
                    temp[index] = bytepair
                    inc embedsDecoded
                    if numberOfOccurencesMet == numberOfOccurences:
                        break
            
            if embedsDecoded == embedding.len:
                # debugecho "stopping decoding at ", embedValue
                break

    result = temp.join("")            

when isMainModule: #test with a random string
    import random, times
    var r = initRand(getTime().toUnix)
    let
        stringLen = rand(r, 150)
        input = newSeqWith(stringLen, rand(r, char)).join()

    assert decode(encode(input)) == input, "This input failed to encode and decode properly : " & input