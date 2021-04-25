import json


in_fname = "bio30_ps.txt"
out_fname = "bio30ps.json"



def edges2json():
    
    result = {}
    edges = []
    
    # file2list each line 1,2... 
    with open( in_fname, 'r' ) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_strs = line.split()
            edge = [ int(i) for i in line_strs ]
            edges.append( edge )
    
    # put list into dict
    result["edges"] = edges
    
    # dict2json and save
    result_str = json.dumps(result)
    with open( out_fname , 'w') as f:
        f.write( result_str )
    



if __name__ == '__main__':
    edges2json()