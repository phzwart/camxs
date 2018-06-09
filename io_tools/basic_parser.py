import sys
import ast


"""
Basic parsing.

Input files are structured according to these rules:

category: key1=value1, key2=value2 ;

the ';' indicates an end point of the line
the ':' indicates a split point in that line 

the key value pairs are returned as dictionairies
"""

def parse_instructions(instructs):
    pairs = instructs.strip().split(',')
    result = {}
    for pair in pairs:
        key,token = pair.split('=')
        key = key.strip()
        p_token = None
        try:
            p_token = ast.literal_eval(token)
        except: pass
        if p_token is None:
            p_token = token
        result[key] = p_token
    return result

def read_and_parse(filename):
    f = open(filename,'r')
    txt = f.read()
    lines = txt.split(';')
    results = {}
    for line in lines:
        if ':' in line:
            keys = line.split(':')
            assert len(keys)==2
            category      = keys[0].strip()
            instructions  = keys[1].strip()
            instructions = parse_instructions(instructions)
            results[category]=instructions
    return results
    
def tst():
    f = open('tst.def','w')
    print >> f,'biryani:cups=4,rice=basmathi,meat=chicken,eggs=7,chili=3.4,spicy=True,  vegetarian=False;'
    f.close()
    results = read_and_parse('tst.def')
    ideal = {'biryani': {'cups': 4, 'meat': 'chicken', 'eggs': 7, 'spicy': True, 'chili': 3.4, 'rice': 'basmathi', 'vegetarian': False}}
    for key in results:
        instructions_1 = results[key]
        instructions_2 = ideal[key] 
        for item in instructions_1.keys():
            val_1 = instructions_1[item]
            val_2 = instructions_2[item]
            assert val_1 == val_2
if __name__ == "__main__":
    tst()
