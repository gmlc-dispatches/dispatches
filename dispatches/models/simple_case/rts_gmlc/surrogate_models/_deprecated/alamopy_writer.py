#I copied the alamopy writer code to do this.  Write out a python function given an alamo model and input labels
def write_alamo_func(mod_res,input_labels, output_name):
    """
    This function writes the file
    - <almname>alm.py
    y=<fname>.f(X)
    preliminary formatting to get the model ready to write
    """
    model = mod_res.split('=')[1]
    model = model + ' '
    tlist = ('sin', 'cos', 'log', 'exp', 'ln')
    for tok in tlist:
        if tok == 'ln':
            model = model.replace(tok, 'np.log')
        else:
            model = model.replace(tok, 'np.' + tok)
    model = model.replace('^', '**')

    with open(output_name + '.py', 'w') as r:
        r.write('import numpy as np\n')
        r.write('def f(*X):\n')
        i = 0
        for label in input_labels:
            r.write('    ' + label + '= X[' + str(i) + ']\n')
            i = i + 1
        r.write('    return ' + model + '\n')
