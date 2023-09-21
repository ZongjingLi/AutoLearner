
numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
yes_or_no = ["yes", "no"]

def num2word(i):
    assert i < 10
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return numbers[i]
    
def copy_dict(d):
    return d if not isinstance(d, dict) else {k: copy_dict(v) for k, v in d.items()}


def head_and_paras(program):
    if len(program) == '' or program == None:return '',[]
    #assert program[-1] == ")", print(program)
    try:
        upper_index = program.index("(")
        assert program[-1] == ")", print(program)
        node_name = program[:upper_index]
        remain = program[upper_index+1:-1]
        args = [];count = 0
        last_start_index = 0

        for i in range(len(remain)):
            e = remain[i]
            if (e == "("):count += 1
            if (e == ")"):count -= 1
            if (count == 0 and e == ","):
                args.append(remain[last_start_index:i])
                last_start_index = i + 1
        args.append(remain[last_start_index:])
        return node_name, args
    except:
        return program, None