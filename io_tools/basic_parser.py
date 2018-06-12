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

def instruction_splitter(txt):
    """
    Splitting the instructions on a comma between key-value pairs.
    Igore commas within brackets.
    """
    split_points = []
    state = 0
    count = 0
    for ii in range( len(txt) ):

        val = txt[ii]
        if val == '(':
            state += 1
        
        if val == ')':
            state -= 1

        if val == ',':
            if state==0:
                split_points.append(count)        
        count += 1 

    results = []
    start_point = 0
    for sp in split_points:
        item = txt[start_point:sp]
        start_point = sp+1
        results.append(item.strip())
    item = txt[start_point:]
    results.append(item.strip())
    return results


def split_parser(token):
   tmp = token.split('(',1)[1]
   tmp = tmp[::-1]
   tmp = tmp.split(')',1)[1]
   tmp = tmp[::-1]
   results = instruction_splitter( tmp  ) 
   x_bit = results[0].strip() 
   y_bit = results[1].strip() 

   x_bit = x_bit.split('(',1)[1]
   x_bit = x_bit[::-1]
   x_bit = x_bit.split(')',1)[1] 
   x_bit = x_bit[::-1]

   y_bit = y_bit.split('(',1)[1]
   y_bit = y_bit[::-1]
   y_bit = y_bit.split(')',1)[1]
   y_bit = y_bit[::-1]

   x_bit = x_bit.split(',')
   y_bit = y_bit.split(',')

   x_bit = slice( int(x_bit[0]), int(x_bit[1]), int(x_bit[2]) )
   y_bit = slice( int(y_bit[0]), int(y_bit[1]), int(y_bit[2]) )

   return (x_bit, y_bit )

specific_key_instructions = { 'split': split_parser }





def parse_instructions(instructs):
    pairs = instruction_splitter(instructs) #instructs.strip().split(',')
    result = {}
    for pair in pairs:
        if len(pair)> 0:
            key,token = pair.split('=')
            key = key.strip()
            p_token = None
            try:
                p_token = ast.literal_eval(token.strip())
            except: pass
            if p_token is None:
                p_token = token
            if specific_key_instructions.has_key( key ):
                p_token = specific_key_instructions[ key ]( p_token  )

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
            # here we have to check if we already have something with that category.
            # if so, we want to create a list of instructions instead

            if results.has_key( category ):
                if type(results[category]) == type([]):
                    results[category].append( instructions  )
                else:
                    results[category] = [ results[category]  ] 
                    results[category].append( instructions  ) 
            else:
                results[category]=instructions
    return results
  
def show_params(params):
    for category in params.keys():
        instructions = params[category]
        # either a single set of instructions or a list
        if type(instructions)==type([]):
            for instr in instructions:
                print( '%s'%('  '+category+(15-len(category))*' '+':'))
                for item in instr.keys():
                    txt = '%s'%('      '+item+' = ' + str(instr[item]) )
                    txt = txt+(40-len(txt))*' '+','
                    print( txt )
                print '  ;'
                print  
        else:
            print( '%s'%('  '+category+(15-len(category))*' '+':'))
            instr = params[ category ]
            for item in instr.keys():
                txt = '%s'%('      '+item+' = ' + str(instr[item]) )
                txt = txt+(40-len(txt))*' '+','
                print( txt )
            print '  ;'
            print
        

 
def tst():
    f = open('tst.def','w')
    print >> f,'biryani:cups=4,rice=basmathi,meat=chicken,eggs=7,chili=3.4,spicy=True,  vegetarian=False; biryani:cups=5,rice=no;'
    f.close()
    results = read_and_parse('tst.def')
    ideal = {'biryani': {'cups': 4, 'meat': 'chicken', 'eggs': 7, 'spicy': True, 'chili': 3.4, 'rice': 'basmathi', 'vegetarian': False}}
    for key in results:
        instructions_1 = results[key][0]
        instructions_2 = ideal[key] 
        for item in instructions_1.keys():
            val_1 = instructions_1[item]
            val_2 = instructions_2[item]
            assert val_1 == val_2
    tst_string = "def: a=((10,20),(30,40)) , hhh=ty, uu=9"
    f = open('tst.def','w')
    print >> f, tst_string
    f.close()
    results = read_and_parse('tst.def')
    assert results['def']['a'][0][0]==10
    assert results['def']['a'][0][1]==20
    assert results['def']['a'][1][0]==30
    assert results['def']['a'][1][1]==40

    tst_string = "ttt: split=( slice(0,120,1), slice(10,90,1) ); "
    f = open('tst.def','w')
    print >> f, tst_string
    f.close() 
    results = read_and_parse('tst.def')

if __name__ == "__main__":
    tst()
