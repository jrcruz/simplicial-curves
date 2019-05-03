import sys
import functools
import operator
import string




def invert(v):
    return [(y, x) for (x, y) in v]


def prune(word):
    result = []
    tmp = ""
    for char in word:
        if char in string.punctuation:
            if len(tmp) > 2:
                result.append("".join(tmp))
                tmp = ""
        else:
            tmp += char.lower()
    if len(tmp) > 2:
        result.append("".join(tmp))
    return result



def main():
    vocab_file = open(sys.argv[1])
    lexicon = dict(invert(enumerate(map(lambda x: x.strip(), vocab_file))))
    # print(lexicon)
    text_file = open(sys.argv[2])

    sequence = []
    for sentence in map(lambda x: x.split(), text_file):
        for word in sentence:
            for pruned in prune(word):
                if pruned in lexicon:
                    sequence.append(str(lexicon[pruned]))
    print(','.join(sequence))



if __name__ == "__main__":
    main()
