import re
import random

# sample all binary str up to max_len
def enumerate_inputs(max_len):
    def genbin(n, bs = ''):
        if n > 0:
            return genbin(n-1, bs + '0') + genbin(n-1, bs + '1')
        else:
            return [bs]
    return sum([genbin(i) for i in range(1,max_len+1)],[])

# priyan sample regex from this grammar
# S -> RP | S RP
# RP -> '[01]' OP | '0' OP | '1' OP
# OP -> '*' | '+' | '{1}' | '{2}'
# some recursive tomfoolery to sample regex
def sample_regex(depth):

    def sample_S(depth,choice=None):
        # u like this hack lmao, putting more 2 to encourage sample longer
        choice = random.choice([1,2,2,2,2]) if choice == None else choice
        if depth == 1:
            return sample_RP(depth-1)
        else:
            if choice == 1:
                return sample_RP(depth-1)
            if choice == 2:
                return ''.join([sample_S(depth-1),sample_RP(depth-1)])
    def sample_RP(depth, choice=None):
        choice = random.choice([1,2,3]) if choice == None else choice        
        if choice == 1:
            return ''.join(['[01]', sample_OP(depth-1)])
        if choice == 2:
            return ''.join(['0', sample_OP(depth-1)])
        if choice == 3:
            return ''.join(['1', sample_OP(depth-1)])
    def sample_OP(depth, choice=None):
        choice = random.choice([1,2,3,4]) if choice == None else choice
        if choice == 1:
            return '*'
        if choice == 2:
            return '+'
        if choice == 3:
            return '{1}'
        if choice == 4:
            return '{2}'

    return sample_S(depth)

# take a regex expression, turn it into a function that
# takes in a string x, return 1 if string match, 0 otherwise
# i.e. to_func('1+0+') : '111000' -> 1
def to_func(regex):
    def func(x):
        return len(re.findall('^'+regex+'$', x))
    return func

# keep sampling fresh functions and add them to the set of functions
# if 2 functions has the same semantix, i.e. \forall x f1(x) = f2(x) 
# we keep the one that is 'shorter'
def sample_unique_functions(all_inputs, n, func_len=3):
    # semantics is literally a vector of outputs for all inputs
    def semantic(f):
        return [f(x) for x in all_inputs]
    semantics_to_regex = dict()

    # pretty much while True, except not.
    for i in range(100000):
        if len(semantics_to_regex) >= n:
            return semantics_to_regex
        regex = sample_regex(func_len)
        regex_semantic = semantic(to_func(regex))
        # turn semantic to a string so we can add it to dict
        regex_semantic = repr(regex_semantic)
        # create a dummy 'long regex' so we can retrieve it later
        if regex_semantic not in semantics_to_regex:
            print ("adding new function ", regex, " total ", len(semantics_to_regex))
            semantics_to_regex[regex_semantic] = 100*'1'
        # retrieve another regex (could be dummy) with same semantics, keep shorter one
        past_regex = semantics_to_regex[regex_semantic]
        regex_to_add = regex if len(regex) < len(past_regex) else past_regex
        semantics_to_regex[regex_semantic] = regex_to_add

    print ("ran out of time guys")
    return semantics_to_regex

if __name__ == '__main__':
    # some short tests
    hii = sample_regex(3)
    print (hii)
    inputs = enumerate_inputs(8)
    for x in inputs:
        print (x, to_func(hii)(x))

    unique_semantics = sample_unique_functions(inputs, 1000)
    print (unique_semantics)