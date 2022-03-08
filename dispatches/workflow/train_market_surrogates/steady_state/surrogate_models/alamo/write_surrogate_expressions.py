import re
from idaes.surrogate.alamopy import AlamoSurrogate

alamo_revenue = AlamoSurrogate.load_from_file(os.path.join('models','alamo_revenue.json'))
alamo_nstartups = AlamoSurrogate.load_from_file(os.path.join('models','alamo_nstartups.json'))
alamo_zones = AlamoSurrogate.load_from_file(os.path.join('models','alamo_zones.json'))

my_labels = ['X1','X2','X3','X4','X5','X6','X7','X8']
alm_labels = alamo_revenue.input_labels()
label_mapping = dict(zip(alm_labels,my_labels))

#################################################
# print the revenue expression
#################################################
model_expression = alamo_revenue._surrogate_expressions['revenue']
for key in label_mapping.keys():
    model_expression = model_expression.replace(key, label_mapping[key])

m = model_expression
all_terms = re.split('(\W\+)|(\W(?<!E)\-)',m)
all_terms[0] = all_terms[0].split("==")[1]

coeffs = []
funcs = []
signs = []
terms = []

#Get signs on coefficients
for term in all_terms:
    if term == None:
        continue
    elif len(term) == 2:
        signs.append(term)
    else:
        terms.append(term)

spl = terms[0].split("*",1)
coeffs.append(round(abs(float(spl[0])),3))
funcs.append(spl[1].strip())

for (i,term) in enumerate(terms[1:]):
    sign = signs[i]
    spl = (sign + term).split("*",1)
    coeffs.append(round(abs(float(spl[0].replace(" ",""))),3))
    funcs.append(spl[1].strip().replace('**','^'))

expression = "revenue == {}{}".format(coeffs[0],funcs[0])
for i in range(1,len(funcs)):
    expression += " {} {}{}".format(signs[i-1],coeffs[i],funcs[i])

print(expression)

#################################################
# print the nstartups expression
#################################################
model_expression = alamo_nstartups._surrogate_expressions['nstartups']
for key in label_mapping.keys():
    model_expression = model_expression.replace(key, label_mapping[key])
m = model_expression
all_terms = re.split('(\W\+)|(\W(?<!E)\-)',m)
all_terms[0] = all_terms[0].split("==")[1]

coeffs = []
funcs = []
signs = []
terms = []

#Get signs on coefficients
for term in all_terms:
    if term == ' ':
        continue 
    elif term == None:
        continue
    elif len(term) == 2:
        signs.append(term)
    else:
        terms.append(term)

spl = terms[0].split("*",1)
coeffs.append(round(abs(float(spl[0])),3))
funcs.append(spl[1].strip())

for (i,term) in enumerate(terms[1:]):
    sign = signs[i]
    spl = (sign + term).split("*",1)
    coeffs.append(round(abs(float(spl[0].replace(" ",""))),3))
    funcs.append(spl[1].strip().replace('**','^'))

expression = "nstartups == {}{}".format(coeffs[0],funcs[0])
for i in range(1,len(funcs)):
    expression += " {} {}{}".format(signs[i-1],coeffs[i],funcs[i])

print(expression)

#################################################
# print zone expressions
#################################################
expressions = []
for j in range(11):
    model_expression = alamo_zones._surrogate_expressions['zone_{}'.format(j)]
    for key in label_mapping.keys():
        model_expression = model_expression.replace(key, label_mapping[key])

    m = model_expression.strip()
    all_terms = re.split('(\W\+)|(\W(?<!E)\-)',m)
    all_terms[0] = all_terms[0].split("==")[1]

    coeffs = []
    funcs = []
    signs = []
    terms = []

    #Get signs on coefficients
    for term in all_terms:
        if term == ' ':
            continue 
        elif term == None:
            continue
        elif len(term) == 2:
            signs.append(term)
        else:
            terms.append(term)

    spl = terms[0].split("*",1)
    coeffs.append(round(abs(float(spl[0])),3))
    funcs.append(spl[1].strip())

    for (i,term) in enumerate(terms[1:]):
        sign = signs[i]
        spl = (sign + term).split("*",1)
        coeffs.append(round(abs(float(spl[0].replace(" ",""))),3))
        if len(spl) > 1: #sometimes there is a constant
            funcs.append(spl[1].strip().replace('**','^'))

    expression = "Y_{{zone{}}} == {} {}{}".format(j,signs[0],coeffs[0],funcs[0])
    for i in range(1,len(coeffs)):
        if i > len(funcs)-1:
            expression += " {} {}".format(signs[i-1],coeffs[i])
        else:
            expression += " {} {}{}".format(signs[i-1],coeffs[i],funcs[i])

    expressions.append(expression)

print(expressions)