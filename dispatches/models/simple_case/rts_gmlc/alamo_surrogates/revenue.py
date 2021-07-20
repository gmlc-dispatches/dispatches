#5 terms, monomial (1,2) , 2-nominal (1,2)
import numpy as np

with open("revenue.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)

xm = model_data["xm"]
xstd = model_data["xstd"]
zm = model_data["zm"]
zstd = model_data["zstd"]    

def f(*X):
    pmax= X[0]
    pmin= X[1]
    ramp_rate= X[2]
    marg_cst= X[3]
    no_load_cst= X[4]
    return  1.3697039265039010480507 * pmax - 0.49029947533655598990165 * pmin - 0.91680507295985447235864 * ramp_rate + 0.28539768199162054984619E-001 * marg_cst - 0.64864939975231392099708E-001 * no_load_cst + 0.59125074000795030393363E-001 * pmax**2 - 0.13488761759743775336950 * pmin**2 + 0.52405422826173565439833E-002 * ramp_rate**2 + 0.25424660114224106877145 * marg_cst**2 + 0.48015530216225302262423E-001 * no_load_cst**2 + 0.14938445017036211526218 * pmax*pmin - 0.92001861841946475095710E-001 * pmax*ramp_rate + 0.95135704722224920248941E-001 * pmax*marg_cst - 0.10147672350323615197976 * pmax*no_load_cst - 0.22776826047681950071500 * pmin*ramp_rate - 0.12376254208917301935511 * pmin*marg_cst + 0.40635005870981941167308E-001 * pmin*no_load_cst - 0.10866703823451814847623 * ramp_rate*marg_cst + 0.94340799247661880078120E-001 * ramp_rate*no_load_cst - 0.93535475673888604508655E-001 * marg_cst*no_load_cst + 0.17659164731119909169665E-001 * (pmax*pmin)**2 + 0.27088763725824728523239E-001 * (pmax*ramp_rate)**2 + 0.74178716514080145216781E-001 * (pmax*marg_cst)**2 - 0.97696627123335846171193E-002 * (pmax*no_load_cst)**2 + 0.25973251881856681579086E-002 * (pmin*ramp_rate)**2 - 0.15909304774349576627746E-001 * (pmin*marg_cst)**2 + 0.31333561576796665343325E-002 * (pmin*no_load_cst)**2 - 0.68250851650735869236009E-001 * (ramp_rate*marg_cst)**2 - 0.29105113599610048490942E-004 * (ramp_rate*no_load_cst)**2 - 0.56811401183926184266237E-001 * (marg_cst*no_load_cst)**2 - 0.25665076181798224252972
