import ConfigParser
import sys
import ast
import io

""" 
I put the output from configparser through some helper classes and create objects 
with names and key / value pairs as laid out in the input file.
I find this a lot easier to work with then the raw config parser objects. 

"""

class section_object(object):
    def __init__(self, items):
        for item in items:
            key = item[0]
            val = item[1].strip()
            try:
                val = ast.literal_eval(val.strip())
            except: pass
            setattr(self, key, val) 

    def as_txt(self):
        txt = """"""
        keys = []
        for key in self.__dict__.keys():
            if '__' not in key:
                keys.append(key)
        keys.sort()
        for key in keys:
            txt += key+'='+str(self.__dict__[key])+' \n'
        return txt


class config_object(object):
    def __init__(self, config):
        for section in config.sections():
            items = config.items(section)
            this_section_object = section_object(items)
            setattr(self, section, this_section_object )

    def as_txt(self):
        txt = """"""
        keys = []
        for key in self.__dict__.keys():
            if '__' not in key:
                keys.append(key)
        #keys.sort()
        for key in keys:
            txt+='[%s]\n'%key
            txt+=self.__dict__[key].as_txt()+' \n'
        return txt

    def show(self,f=None):
        if f is None:
            f = sys.stdout
        print >> f, self.as_txt()


def read_and_parse(inputs,defaults=None):
    # make config object
    config = ConfigParser.ConfigParser()

    # not sure how the defaults work
    if defaults is not None:
        default_config = None
        if type(defaults) is str:
            default_config = ConfigParser.ConfigParser()
            default_config.readfp(io.BytesIO(defaults))
        if type(defaults) is file:
            default_config = ConfigParser.ConfigParser()
            default_config.readfp(defaults)
        # if this aint a config object, we're f-ed anyway
        config._sections = default_config._sections

    if inputs is None:
        co = config_object( default_config )
        co.show()
        raise SystemExit('##--- No inputs provide, use template shown above ---##') 


    if type(inputs) is type(""""""):
        inputs = io.BytesIO(inputs) 
    # update the config object accoring to the specified input
    config.readfp(inputs)

    # build the object.
    co = config_object( config )
    return co

def tst():
    default_instructions = """    
[data]
experiment   = None
run          = None
index_start  = 0
index_stop   = 500
index_stride = 1 

[output]
filename = output.h5
comments = "No comments"
"""
    
    instructions = """
[data]
experiment   = amox26916
run          = 56
index_start  = 0
index_stop   = 5000

[output]
filename = output_59.h5
comments = "No soup for you!"
ooops = 9
""" 

    obj = read_and_parse(instructions, default_instructions)

    instructions="""
[stuff]
coordinates = (1,4) 
"""
    obj = read_and_parse(io.BytesIO(instructions))
    assert obj.stuff.coordinates[0] == 1
    assert obj.stuff.coordinates[1] == 4
    obj.show()
    print 'OK'


if __name__ == "__main__":
    tst()
