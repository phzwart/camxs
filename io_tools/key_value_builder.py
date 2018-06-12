import ast

function_pairing = {
                       'split' : slice,
                   } 


def split_parser(token):
   tmp = token.split('(',1)[1]
   tmp = tmp[::-1]
   tmp = tmp.split(')',1)[1]
   tmp = tmp[::-1]
   x_part, y_part = 


def parse(key, token):
    value = None

    if function_pairing.has_key( key ):
       None 
    else:
       value = ast.ast.literal_eval(token.strip())
    
    return token

def tst(txt):
    split_parser(txt)
    

if __name__ == "__main__":
    txt = "( slice(0,120,1), slice(10,90,1) ) oooo"
    tst(txt)


