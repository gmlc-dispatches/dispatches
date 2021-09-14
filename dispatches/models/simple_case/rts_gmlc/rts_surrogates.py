import pickle
#Revenue surrogate using only 5 terms
with open("alamo_surrogates/revenue_5_terms.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm_rev = model_data["xm"]
xstd_rev = model_data["xstd"]
zm_rev = model_data["zm"]
zstd_rev = model_data["zstd"]
def revenue_rule_5_terms(m):
    pmax = (m.pmax - xm_rev[0])/xstd_rev[0]
    pmin = (m.pmin - xm_rev[1])/xstd_rev[1]
    ramp_rate = (m.ramp_rate - xm_rev[2])/xstd_rev[2]
    marg_cst = (m.marg_cst - xm_rev[3])/xstd_rev[3]
    no_load_cst = (m.no_load_cst - xm_rev[4])/xstd_rev[4]
    z =  1.3697039265039010480507 * pmax - 0.49029947533655598990165 * pmin - 0.91680507295985447235864 * ramp_rate + 0.28539768199162054984619E-001 * marg_cst - 0.64864939975231392099708E-001 * no_load_cst + 0.59125074000795030393363E-001 * pmax**2 - 0.13488761759743775336950 * pmin**2 + 0.52405422826173565439833E-002 * ramp_rate**2 + 0.25424660114224106877145 * marg_cst**2 + 0.48015530216225302262423E-001 * no_load_cst**2 + 0.14938445017036211526218 * pmax*pmin - 0.92001861841946475095710E-001 * pmax*ramp_rate + 0.95135704722224920248941E-001 * pmax*marg_cst - 0.10147672350323615197976 * pmax*no_load_cst - 0.22776826047681950071500 * pmin*ramp_rate - 0.12376254208917301935511 * pmin*marg_cst + 0.40635005870981941167308E-001 * pmin*no_load_cst - 0.10866703823451814847623 * ramp_rate*marg_cst + 0.94340799247661880078120E-001 * ramp_rate*no_load_cst - 0.93535475673888604508655E-001 * marg_cst*no_load_cst + 0.17659164731119909169665E-001 * (pmax*pmin)**2 + 0.27088763725824728523239E-001 * (pmax*ramp_rate)**2 + 0.74178716514080145216781E-001 * (pmax*marg_cst)**2 - 0.97696627123335846171193E-002 * (pmax*no_load_cst)**2 + 0.25973251881856681579086E-002 * (pmin*ramp_rate)**2 - 0.15909304774349576627746E-001 * (pmin*marg_cst)**2 + 0.31333561576796665343325E-002 * (pmin*no_load_cst)**2 - 0.68250851650735869236009E-001 * (ramp_rate*marg_cst)**2 - 0.29105113599610048490942E-004 * (ramp_rate*no_load_cst)**2 - 0.56811401183926184266237E-001 * (marg_cst*no_load_cst)**2 - 0.25665076181798224252972

    z_unscale = z*zstd_rev + zm_rev
    return z_unscale

#revenue surrogate using all terms
with open("alamo_surrogates/revenue_all_terms.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm_rev_all = model_data["xm"]
xstd_rev_all = model_data["xstd"]
zm_rev_all = model_data["zm"]
zstd_rev_all = model_data["zstd"]
def revenue_rule_all_terms(m):
    pmax = (m.pmax - xm_rev_all[0])/xstd_rev_all[0]
    pmin = (m.pmin - xm_rev_all[1])/xstd_rev_all[1]
    ramp_rate = (m.ramp_rate - xm_rev_all[2])/xstd_rev_all[2]
    min_up_time = (m.min_up_time - xm_rev_all[3])/xstd_rev_all[3]
    min_down_time = (m.min_down_time  - xm_rev_all[4])/xstd_rev_all[4]
    marg_cst = (m.marg_cst - xm_rev_all[5])/xstd_rev_all[5]
    no_load_cst = (m.no_load_cst - xm_rev_all[6])/xstd_rev_all[6]
    st_time_hot = (m.st_time_hot - xm_rev_all[7])/xstd_rev_all[7]
    st_time_warm = (m.st_time_warm - xm_rev_all[8])/xstd_rev_all[8]
    st_time_cold = (m.st_time_cold - xm_rev_all[9])/xstd_rev_all[9]
    st_cst_hot = (m.st_cst_hot - xm_rev_all[10])/xstd_rev_all[10]
    st_cst_warm = (m.st_cst_warm - xm_rev_all[11])/xstd_rev_all[11]
    st_cst_cold = (m.st_cst_cold - xm_rev_all[12])/xstd_rev_all[12]

    z =  1.3671359660782966827242 * pmax - 0.49150401227448226038064 * pmin - 0.91633462273622989791022 * ramp_rate + 0.60916643924719994507289E-002 * min_up_time + 0.44530645110430876892904E-002 * min_down_time + 0.20937642703956815815047E-001 * marg_cst - 0.64507761820002590402723E-001 * no_load_cst + 0.56979691765289941507433E-001 * st_time_hot - 0.16451322348143015278366E-001 * st_time_warm - 0.57405481161860505423533E-001 * st_time_cold - 0.21284304802386029564776E-002 * st_cst_hot - 0.27645091525231112183913E-001 * st_cst_warm + 0.35130400782158878458805E-001 * pmax**2 - 0.12619088230895586510982 * pmin**2 + 0.16277706834075179875843E-001 * ramp_rate**2 - 0.10218145483308852666804E-001 * min_up_time**2 - 0.56059553177327569109534E-002 * min_down_time**2 + 0.11115678009971791118105 * marg_cst**2 + 0.36798806545966421255311E-001 * no_load_cst**2 - 0.23577726922012040566834 * st_time_hot**2 + 0.14514917183903885966600 * pmax*pmin - 0.92641659785577881724983E-001 * pmax*ramp_rate + 0.91265707517018082595150E-001 * pmax*marg_cst - 0.10073895527872628319344 * pmax*no_load_cst - 0.45287439029696306691530E-001 * pmax*st_time_hot + 0.12592941430926457568873E-001 * pmax*st_time_warm - 0.12930848856311622299686E-001 * pmax*st_time_cold - 0.25111479542852210150583E-001 * pmax*st_cst_hot - 0.22801562099443614672900 * pmin*ramp_rate - 0.12610052538897567608878 * pmin*marg_cst + 0.42142515444356661025171E-001 * pmin*no_load_cst - 0.11278627160379269597085E-002 * pmin*st_time_hot + 0.16269823091684291332948E-001 * pmin*st_time_warm + 0.98432231288233234395291E-002 * pmin*st_time_cold - 0.47294457720659845850752E-002 * pmin*st_cst_hot - 0.10683261184184765502092 * ramp_rate*marg_cst + 0.93961020038044046343018E-001 * ramp_rate*no_load_cst + 0.71580937568893263089898E-001 * ramp_rate*st_time_hot - 0.35607288542668230624244E-001 * ramp_rate*st_time_warm + 0.80861390699648507535136E-002 * ramp_rate*st_time_cold + 0.43274080993659869154300E-001 * ramp_rate*st_cst_hot + 0.14234376173873389617719E-001 * min_up_time*marg_cst + 0.55179172642412536650691E-002 * min_up_time*no_load_cst + 0.22704879992383503357900E-002 * min_up_time*st_time_hot + 0.38597008872397410746136E-002 * min_up_time*st_time_cold - 0.12513218466922474206293E-001 * min_down_time*marg_cst + 0.18284564352283511068364E-001 * min_down_time*st_cst_hot - 0.91122671063783008960080E-001 * marg_cst*no_load_cst - 0.13785625482364139218761E-001 * marg_cst*st_time_hot - 0.36636523588944539669976E-002 * marg_cst*st_time_warm - 0.15687752072251681320636E-002 * marg_cst*st_time_cold - 0.30581019824640294502149E-001 * marg_cst*st_cst_hot + 0.22145467019937326372259E-002 * no_load_cst*st_time_hot - 0.12937850436591491129490E-001 * no_load_cst*st_time_warm + 0.19730305898492467991945E-001 * (pmax*pmin)**2 + 0.27132667025413566286307E-001 * (pmax*ramp_rate)**2 + 0.74432803856587737012518E-001 * (pmax*marg_cst)**2 - 0.76744978421927796674584E-002 * (pmax*no_load_cst)**2 + 0.18663244630402239798705E-001 * (pmax*st_time_hot)**2 - 0.22200953937263501886124E-002 * (pmin*min_down_time)**2 - 0.20125927290930396146296E-001 * (pmin*marg_cst)**2 - 0.71534893526416483758301E-002 * (pmin*st_time_hot)**2 + 0.95068557982725277605285E-002 * (pmin*st_cst_hot)**2 - 0.66630425019571601352730E-001 * (ramp_rate*marg_cst)**2 - 0.10183372912370394075543E-001 * (ramp_rate*st_time_hot)**2 - 0.10806615375335215049196E-002 * (min_up_time*marg_cst)**2 + 0.97523362541790965113409E-002 * (min_up_time*st_time_hot)**2 - 0.67952421247023933256748E-002 * (min_down_time*marg_cst)**2 + 0.88193464911783189807970E-002 * (min_down_time*st_time_hot)**2 - 0.39902046155824752449170E-001 * (marg_cst*no_load_cst)**2 + 0.13880311321374966260223 * (marg_cst*st_time_hot)**2 + 0.73463018561151010796251E-002 * (marg_cst*st_time_warm)**2 + 0.32115952056451416152250E-001 * (marg_cst*st_time_cold)**2 - 0.60882899610359232644985E-001 * (marg_cst*st_cst_hot)**2

    z_unscale = z*zstd_rev + zm_rev
    return z_unscale

with open("alamo_surrogates/hours_zone_0.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm0 = model_data["xm"]
xstd0 = model_data["xstd"]
zm0 = model_data["zm"]
zstd0 = model_data["zstd"]

def hours_zone_0(m):
    pmax = (m.pmax - xm0[0])/xstd0[0]
    pmin = (m.pmin - xm0[1])/xstd0[1]
    ramp_rate = (m.ramp_rate - xm0[2])/xstd0[2]
    min_up_time = (m.min_up_time - xm0[3])/xstd0[3]
    min_down_time = (m.min_down_time  - xm0[4])/xstd0[4]
    marg_cst = (m.marg_cst - xm0[5])/xstd0[5]
    no_load_cst = (m.no_load_cst - xm0[6])/xstd0[6]
    st_time_hot = (m.st_time_hot - xm0[7])/xstd0[7]
    st_time_warm = (m.st_time_warm - xm0[8])/xstd0[8]
    st_time_cold = (m.st_time_cold - xm0[9])/xstd0[9]
    st_cst_hot = (m.st_cst_hot - xm0[10])/xstd0[10]
    st_cst_warm = (m.st_cst_warm - xm0[11])/xstd0[11]
    st_cst_cold = (m.st_cst_cold - xm0[12])/xstd0[12]

    z = - 0.24862380577373788259621 * pmax + 0.17327062432224631799427E-001 * pmin + 0.23685716421390515251666 * ramp_rate + 0.45055290936102700138921E-001 * min_up_time + 0.70752743273035684223871E-001 * min_down_time + 0.42112674379804854174481 * marg_cst - 0.85107409627752775294063E-001 * no_load_cst + 0.16759062172436911541951 * pmax**2 + 0.97263916438612707526801E-001 * pmin**2 - 0.30952195890901514074844E-002 * ramp_rate**2 - 0.60052797879169082795325E-001 * min_up_time**2 + 0.19414265862495030945389 * min_down_time**2 + 0.35708514256534668040999 * marg_cst**2 - 0.32053013480223956077619E-001 * no_load_cst**2 - 0.47829670642563310023476 * st_time_hot**2 - 0.19090177426903973123551 * pmax*pmin - 0.24264801202535335455934 * pmax*ramp_rate - 0.68412166961012137872400E-002 * pmax*min_up_time - 0.10308745689925102526074E-001 * pmax*min_down_time + 0.11792333541566309140780 * pmax*marg_cst + 0.20057572015445949281665E-001 * pmax*no_load_cst + 0.38989077949801276279373E-001 * pmax*st_time_hot + 0.71604196040067819885311E-003 * pmax*st_time_warm - 0.86180812399662594120581E-003 * pmax*st_time_cold - 0.64117653404630950420007E-001 * pmax*st_cst_warm + 0.15141356569987446389902 * pmin*ramp_rate + 0.28707766409980878252739E-001 * pmin*min_up_time + 0.37223939515197521088474E-001 * pmin*min_down_time - 0.24416178672367350177552 * pmin*marg_cst - 0.83071400418676000221296E-001 * pmin*no_load_cst - 0.29938535062524063157241 * pmin*st_time_hot + 0.15711282640199480953314 * pmin*st_time_warm - 0.60886412495017125379171E-001 * pmin*st_time_cold + 0.14745583224904151853352 * pmin*st_cst_hot - 0.17966397836370582435928E-001 * ramp_rate*marg_cst - 0.78339496415421164265958E-001 * ramp_rate*st_time_hot + 0.33963564998538177397425E-001 * ramp_rate*st_time_warm - 0.20507444249751781428781E-001 * ramp_rate*st_time_cold + 0.44986817916690104157684E-001 * ramp_rate*st_cst_hot + 0.55098264616900656520659E-001 * min_up_time*min_down_time + 0.27550607534113688568134E-001 * min_up_time*marg_cst + 0.32739458400345151922739E-001 * min_up_time*st_time_hot - 0.35534326419119653672762E-001 * min_up_time*st_time_warm - 0.68436595992225094861605E-002 * min_up_time*st_time_cold + 0.30243391511432234780576E-001 * min_up_time*st_cst_hot + 0.63484502946734369666082E-002 * min_down_time*marg_cst + 0.45140222164158930184819E-001 * min_down_time*st_time_hot + 0.89015304665737510214640E-001 * min_down_time*st_time_warm + 0.99802611085874959329090E-001 * min_down_time*st_time_cold - 0.29017846889920573261179 * min_down_time*st_cst_hot - 0.93248246755490829529300E-001 * marg_cst*no_load_cst - 0.86581713196571361002007E-001 * marg_cst*st_time_hot + 0.35794832507611978877904E-001 * marg_cst*st_time_warm - 0.21539741913674768925002E-001 * marg_cst*st_time_cold + 0.57747656170179843815315E-001 * marg_cst*st_cst_hot + 0.29721922660720288822400E-001 * no_load_cst*st_time_hot - 0.20354378858134149904435E-001 * no_load_cst*st_time_warm + 0.75355827882367353021120E-002 * no_load_cst*st_time_cold - 0.12629407013818486765766E-001 * (pmax*pmin)**2 + 0.18431910148852402760167E-002 * (pmax*ramp_rate)**2 - 0.25544047267575340637302E-001 * (pmax*marg_cst)**2 + 0.11777744897696418649446E-001 * (pmax*st_time_hot)**2 + 0.24379596146921270194419E-001 * (pmax*st_time_warm)**2 + 0.21834463061770871539213E-001 * (pmax*st_time_cold)**2 - 0.57041423507817122506847E-001 * (pmax*st_cst_hot)**2 + 0.30720904627244957629806E-001 * (pmin*ramp_rate)**2 - 0.47377559863026781628026E-002 * (pmin*min_up_time)**2 - 0.57069796067566944119953E-002 * (pmin*min_down_time)**2 - 0.10571686480617776757174E-002 * (pmin*marg_cst)**2 + 0.17857735864080622606442E-001 * (pmin*no_load_cst)**2 + 0.30062806391480444492825E-002 * (pmin*st_time_hot)**2 - 0.72644086105194308200517E-001 * (pmin*st_time_warm)**2 - 0.59235638328552317477538E-001 * (pmin*st_time_cold)**2 + 0.16562684289073240084811 * (pmin*st_cst_hot)**2 + 0.44874076797134142269297E-001 * (ramp_rate*marg_cst)**2 + 0.55796889665310125255848E-002 * (ramp_rate*no_load_cst)**2 - 0.83628002630860114041678E-002 * (ramp_rate*st_time_hot)**2 - 0.18051845142650086578628E-001 * (ramp_rate*st_time_warm)**2 - 0.16506236640101777657375E-001 * (ramp_rate*st_time_cold)**2 + 0.44947031941919263209329E-001 * (ramp_rate*st_cst_hot)**2 - 0.28565035248769000963964E-001 * (min_up_time*min_down_time)**2 + 0.30891021044325391725627E-001 * (min_up_time*marg_cst)**2 + 0.24762170240600324400138E-001 * (min_up_time*st_time_hot)**2 - 0.61558194180567496825440E-002 * (min_up_time*st_time_warm)**2 + 0.15695767434182557320543E-001 * (min_up_time*st_cst_hot)**2 - 0.44703186252772345379847E-002 * (min_down_time*marg_cst)**2 - 0.79017960468237086191223E-001 * (min_down_time*st_time_hot)**2 - 0.20291334295536655324410E-001 * (min_down_time*st_time_warm)**2 - 0.29908034948573108957603E-001 * (min_down_time*st_time_cold)**2 - 0.11190169266645722673248E-001 * (marg_cst*no_load_cst)**2 + 0.27529812059783463629170E-001 * (marg_cst*st_time_hot)**2 + 0.61539753949013042619298E-002 * (marg_cst*st_time_warm)**2 + 0.13030854601293866829037E-001 * (marg_cst*st_time_cold)**2 - 0.65475920208801566269052E-001 * (marg_cst*st_cst_hot)**2 + 0.15394070274716510973723E-001 * (no_load_cst*st_time_hot)**2

    z_unscale = z*zstd0 + zm0
    return z_unscale

with open("alamo_surrogates/hours_zone_1.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm1 = model_data["xm"]
xstd1 = model_data["xstd"]
zm1 = model_data["zm"]
zstd1 = model_data["zstd"]
def hours_zone_1(m):
    pmax = (m.pmax - xm1[0])/xstd1[0]
    pmin = (m.pmin - xm1[1])/xstd1[1]
    ramp_rate = (m.ramp_rate - xm1[2])/xstd1[2]
    min_up_time = (m.min_up_time - xm1[3])/xstd1[3]
    min_down_time = (m.min_down_time  - xm1[4])/xstd1[4]
    marg_cst = (m.marg_cst - xm1[5])/xstd1[5]
    no_load_cst = (m.no_load_cst - xm1[6])/xstd1[6]
    st_time_hot = (m.st_time_hot - xm1[7])/xstd1[7]
    st_time_warm = (m.st_time_warm - xm1[8])/xstd1[8]
    st_time_cold = (m.st_time_cold - xm1[9])/xstd1[9]
    st_cst_hot = (m.st_cst_hot - xm1[10])/xstd1[10]
    st_cst_warm = (m.st_cst_warm - xm1[11])/xstd1[11]
    st_cst_cold = (m.st_cst_cold - xm1[12])/xstd1[12]

    z = 0.73637873016076316190492 * pmax - 0.44271096830202721905678 * pmin + 0.10423129065789450697910 * ramp_rate + 0.29361230558311742638855E-001 * min_up_time + 0.34161093244243413702410E-001 * min_down_time + 0.13731907268748275163794 * marg_cst - 0.16519181313399602939462 * no_load_cst + 0.57996223156719234337331E-001 * pmax**2 + 0.29647887630074802600699E-001 * pmin**2 + 0.67191634438725583722274E-001 * ramp_rate**2 - 0.20449558330082464457922E-001 * min_up_time**2 + 0.50077011889861092197584E-001 * min_down_time**2 - 0.66600227280443596855619E-001 * marg_cst**2 - 0.77664902828839990633902E-001 * no_load_cst**2 + 0.28108260838998859465487E-001 * pmax*pmin - 0.11929206290728180950289 * pmax*ramp_rate + 0.17468630506575270194825E-001 * pmax*min_up_time + 0.18812442154179906395051 * pmax*marg_cst - 0.65970684436877868872529E-001 * pmax*no_load_cst - 0.78997249905357683208429E-001 * pmax*st_time_hot + 0.42312134007863801132832E-001 * pmax*st_time_warm - 0.93950516303464900319531E-002 * pmax*st_time_cold + 0.70707224868498730341315E-001 * pmin*ramp_rate - 0.98222298168389662875732E-002 * pmin*min_up_time - 0.75526837477457222508326E-002 * pmin*min_down_time - 0.21957556950761017433571 * pmin*marg_cst + 0.42336929641152057368636E-001 * pmin*no_load_cst + 0.57425262577566249522221E-001 * pmin*st_time_hot - 0.32508422898256589261834E-001 * pmin*st_time_warm + 0.84628560841247934404263E-002 * pmin*st_time_cold - 0.65194982749013308254149E-001 * ramp_rate*marg_cst + 0.13670983120745963841636E-002 * ramp_rate*st_time_hot - 0.11199718531501548501872E-001 * ramp_rate*st_time_warm - 0.15914744390631351683707E-001 * ramp_rate*st_time_cold + 0.40277204054550624912068E-001 * ramp_rate*st_cst_hot + 0.36434172146717104523450E-001 * min_up_time*marg_cst + 0.32758303722463695539791E-001 * min_up_time*no_load_cst + 0.21098126373485272255781E-001 * min_up_time*st_time_hot - 0.11817346238985790632392E-001 * min_up_time*st_time_warm + 0.74187549727449172282112E-002 * min_up_time*st_time_cold - 0.18780539245621924449026E-001 * min_up_time*st_cst_hot - 0.37995499109109969609888E-002 * min_down_time*marg_cst + 0.93291107175692538494571E-002 * min_down_time*no_load_cst + 0.77308828384659439292292E-001 * min_down_time*st_time_hot - 0.21305187853581031981465E-001 * min_down_time*st_time_warm + 0.28674345624316508601703E-001 * min_down_time*st_time_cold - 0.47679986028343154802478E-001 * min_down_time*st_cst_hot - 0.15776576745849052452186 * marg_cst*no_load_cst - 0.72191556704986212245068E-001 * marg_cst*st_time_hot - 0.26652771112984578738558E-001 * marg_cst*st_time_warm - 0.10199696764865992873461E-001 * marg_cst*st_time_cold + 0.10320948202671681448450 * marg_cst*st_cst_warm - 0.10434783085972634819605 * no_load_cst*st_time_hot + 0.23586376608539943305898E-001 * no_load_cst*st_time_warm - 0.39454398416863718790371E-001 * no_load_cst*st_time_cold + 0.90779922056779122074843E-001 * no_load_cst*st_cst_hot - 0.54527712687858359563720E-001 * (pmax*pmin)**2 + 0.24312656719654261727737E-001 * (pmax*marg_cst)**2 + 0.60396148919610349870002E-002 * (pmin*no_load_cst)**2 - 0.10343346758486164724961E-002 * (pmin*st_time_warm)**2 + 0.18741867285554072303233E-001 * (pmin*st_cst_hot)**2 - 0.35701382877712261633418E-002 * (ramp_rate*marg_cst)**2 + 0.14419971713652959821594E-001 * (ramp_rate*no_load_cst)**2 - 0.10584203416548795817498E-001 * (ramp_rate*st_time_hot)**2 - 0.62345425776635116782431E-002 * (ramp_rate*st_cst_cold)**2 - 0.11269276813934298378528E-001 * (min_up_time*min_down_time)**2 + 0.13828583748567368008620E-001 * (min_up_time*marg_cst)**2 - 0.79839517601936419582964E-002 * (min_up_time*no_load_cst)**2 + 0.98021018143731163813070E-002 * (min_up_time*st_time_hot)**2 + 0.12606292066485667510700E-001 * (min_up_time*st_cst_cold)**2 - 0.10222452831164692554911E-001 * (min_down_time*marg_cst)**2 - 0.12808350344021021335883E-001 * (min_down_time*st_cst_cold)**2 + 0.22049859977130173427362E-001 * (marg_cst*st_time_hot)**2 - 0.15121740175452098864373E-001 * (marg_cst*st_time_warm)**2 - 0.37397242768599809475960E-002 * (marg_cst*st_time_cold)**2 + 0.26267327309829394815788E-001 * (marg_cst*st_cst_hot)**2 + 0.64419543218616495683371E-001 * (no_load_cst*st_time_hot)**2 - 0.13630773519587847247836E-001 * (no_load_cst*st_time_warm)**2 + 0.22041658465819832790622E-001 * (no_load_cst*st_cst_hot)**2

    z_unscale = z*zstd1 + zm1
    return z_unscale

with open("alamo_surrogates/hours_zone_2.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm2 = model_data["xm"]
xstd2 = model_data["xstd"]
zm2 = model_data["zm"]
zstd2 = model_data["zstd"]
def hours_zone_2(m):
    pmax = (m.pmax - xm2[0])/xstd2[0]
    pmin = (m.pmin - xm2[1])/xstd2[1]
    ramp_rate = (m.ramp_rate - xm2[2])/xstd2[2]
    min_up_time = (m.min_up_time - xm2[3])/xstd2[3]
    min_down_time = (m.min_down_time  - xm2[4])/xstd2[4]
    marg_cst = (m.marg_cst - xm2[5])/xstd2[5]
    no_load_cst = (m.no_load_cst - xm2[6])/xstd2[6]
    st_time_hot = (m.st_time_hot - xm2[7])/xstd2[7]
    st_time_warm = (m.st_time_warm - xm2[8])/xstd2[8]
    st_time_cold = (m.st_time_cold - xm2[9])/xstd2[9]
    st_cst_hot = (m.st_cst_hot - xm2[10])/xstd2[10]
    st_cst_warm = (m.st_cst_warm - xm2[11])/xstd2[11]
    st_cst_cold = (m.st_cst_cold - xm2[12])/xstd2[12]

    z = 0.78804915690342236533894 * pmax - 0.48923838864473406795597 * pmin + 0.74713028033437725583532E-001 * ramp_rate + 0.13544317650722840357114E-001 * min_up_time + 0.35315280575944574259495E-001 * min_down_time + 0.11077014609627104657630 * marg_cst - 0.15166997056119810305397 * no_load_cst - 0.73791814445500647501319E-001 * pmax**2 - 0.61247401006383241084396E-001 * pmin**2 - 0.90443914844343484271683E-001 * ramp_rate**2 - 0.20750035776580463137142E-001 * min_up_time**2 + 0.24828624308924233943241E-001 * marg_cst**2 - 0.30417249148305441214246E-001 * no_load_cst**2 + 0.11693399686746776267032 * pmax*pmin + 0.20822889357629875695110 * pmax*ramp_rate + 0.12559228144062923454571E-001 * pmax*min_up_time + 0.18084107744953142882238 * pmax*marg_cst - 0.39108326948002480882849E-001 * pmax*no_load_cst - 0.40417536055888769741529E-001 * pmax*st_time_hot + 0.18185965825056819855643E-001 * pmax*st_time_warm - 0.13180753078393836474902 * pmin*ramp_rate - 0.22050790888389687136040E-002 * pmin*min_up_time - 0.14200712478550769207741E-001 * pmin*min_down_time - 0.22848124561989063829870 * pmin*marg_cst + 0.24549447334676607890280E-001 * pmin*no_load_cst + 0.37564981948630081498575E-001 * pmin*st_time_hot - 0.17905074238715529971744E-001 * pmin*st_time_warm - 0.87224150206798134465913E-001 * ramp_rate*marg_cst - 0.19769409499292317500263E-001 * ramp_rate*no_load_cst + 0.48013485556254982891677E-002 * ramp_rate*st_time_hot - 0.36471756337179084273004E-001 * ramp_rate*st_time_warm - 0.94354567304807835520508E-002 * ramp_rate*st_time_cold + 0.60152328203161936626131E-001 * ramp_rate*st_cst_warm + 0.33533171479388716729186E-001 * min_up_time*marg_cst + 0.29640476473426248527065E-001 * min_up_time*no_load_cst - 0.70064030319086376477622E-002 * min_up_time*st_time_hot + 0.37887547861144392491450E-001 * min_up_time*st_time_warm + 0.14677593958514794508785E-001 * min_up_time*st_time_cold - 0.53307411391531139832622E-001 * min_up_time*st_cst_warm + 0.94399239418708636578659E-002 * min_down_time*no_load_cst + 0.54272614688762509105313E-001 * min_down_time*st_time_hot - 0.30226746868516968930196E-001 * min_down_time*st_time_warm - 0.15669557623565488757578 * marg_cst*no_load_cst - 0.52558684766518266873181E-001 * marg_cst*st_time_hot - 0.61364130235599063933893E-001 * marg_cst*st_time_warm - 0.41780955565231391035663E-001 * marg_cst*st_cst_hot + 0.16860063637896932631222 * marg_cst*st_cst_warm - 0.94235158774838026496390E-001 * no_load_cst*st_time_hot + 0.25194739672152320630882E-001 * no_load_cst*st_time_warm - 0.33491882247556246932074E-001 * no_load_cst*st_time_cold + 0.78541235436124373636702E-001 * no_load_cst*st_cst_hot + 0.14158175327708183779962E-001 * (pmax*pmin)**2 - 0.38780177839197300937446E-001 * (pmax*ramp_rate)**2 - 0.13404649541458299455421E-001 * (pmax*marg_cst)**2 - 0.92898292840893088168075E-002 * (pmax*no_load_cst)**2 + 0.16448673974010404041923E-001 * (pmin*ramp_rate)**2 + 0.75357413451521537001154E-002 * (pmin*marg_cst)**2 + 0.92896732972527901184279E-002 * (pmin*st_cst_cold)**2 + 0.10131900661673050659251E-001 * (ramp_rate*no_load_cst)**2 - 0.12737430760911775534661E-001 * (ramp_rate*st_time_hot)**2 + 0.72659644970264379917890E-002 * (min_up_time*marg_cst)**2 + 0.22319246660986991753761E-001 * (min_up_time*st_time_hot)**2 - 0.74331141830033834946856E-002 * (min_down_time*marg_cst)**2 - 0.34629747743041765055405E-002 * (min_down_time*no_load_cst)**2 - 0.22651648801401309818448E-001 * (marg_cst*no_load_cst)**2 - 0.16294311326562744418611E-001 * (marg_cst*st_time_hot)**2 - 0.13838690275246211952576E-001 * (marg_cst*st_time_warm)**2 - 0.91432743591486325546169E-002 * (marg_cst*st_time_cold)**2 + 0.44061526914870835502924E-001 * (marg_cst*st_cst_hot)**2 + 0.59197648607355393690632E-001 * (no_load_cst*st_time_hot)**2 - 0.14151341289041835178053E-001 * (no_load_cst*st_time_warm)**2

    z_unscale = z*zstd2 + zm2
    return z_unscale

with open("alamo_surrogates/hours_zone_3.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm3 = model_data["xm"]
xstd3 = model_data["xstd"]
zm3 = model_data["zm"]
zstd3 = model_data["zstd"]

def hours_zone_3(m):
    pmax = (m.pmax - xm3[0])/xstd3[0]
    pmin = (m.pmin - xm3[1])/xstd3[1]
    ramp_rate = (m.ramp_rate - xm3[2])/xstd3[2]
    min_up_time = (m.min_up_time - xm3[3])/xstd3[3]
    min_down_time = (m.min_down_time  - xm3[4])/xstd3[4]
    marg_cst = (m.marg_cst - xm3[5])/xstd3[5]
    no_load_cst = (m.no_load_cst - xm3[6])/xstd3[6]
    st_time_hot = (m.st_time_hot - xm3[7])/xstd3[7]
    st_time_warm = (m.st_time_warm - xm3[8])/xstd3[8]
    st_time_cold = (m.st_time_cold - xm3[9])/xstd3[9]
    st_cst_hot = (m.st_cst_hot - xm3[10])/xstd3[10]
    st_cst_warm = (m.st_cst_warm - xm3[11])/xstd3[11]
    st_cst_cold = (m.st_cst_cold - xm3[12])/xstd3[12]

    z = 0.71273976966567265112218 * pmax - 0.39745173548847384514815 * pmin - 0.32840510691883267879732 * ramp_rate - 0.30693388760841180257222E-002 * min_up_time - 0.64549394889441724654233E-001 * min_down_time + 0.12504869243228181052707 * marg_cst - 0.30565822608836473728289 * st_time_hot + 0.19079695949027908330464 * st_time_warm + 0.26013975356374202663190 * st_time_cold - 0.93888667435570938302192 * pmax**2 - 0.44911535532127305758721 * pmin**2 - 1.4980564549150749975581 * ramp_rate**2 - 0.15320123201520829264721E-002 * min_up_time**2 - 0.21739938162966217227723E-001 * min_down_time**2 + 0.15887687715466600191228E-001 * marg_cst**2 + 0.55000965920230338035424 * st_time_hot**2 - 0.54891123505346979538633E-001 * st_time_cold**2 + 1.2830321700099318604771 * pmax*pmin + 2.4686292333214363559080 * pmax*ramp_rate - 0.43130052611650988858560E-001 * pmax*min_up_time - 0.43853278045985710775767E-001 * pmax*min_down_time + 0.16203660730224683783618 * pmax*marg_cst + 0.93617390554999416196758E-001 * pmax*no_load_cst + 0.34529660788819621641821 * pmax*st_time_hot - 0.17741221593106584752952 * pmax*st_time_warm + 0.82394633690457519392680E-001 * pmax*st_time_cold - 0.16654479435646332707854 * pmax*st_cst_hot - 1.6600288651383319216137 * pmin*ramp_rate + 0.36891373272261794580285E-001 * pmin*min_up_time - 0.11075519386950706535178 * pmin*marg_cst - 0.67507581003326891377903E-001 * pmin*no_load_cst - 0.17347174915842897280349 * pmin*st_time_hot + 0.94903885823527184273374E-001 * pmin*st_time_warm - 0.34977890261985083997232E-001 * pmin*st_time_cold + 0.62110501303729595545811E-001 * pmin*st_cst_hot + 0.56960090317472324150661E-001 * ramp_rate*min_up_time + 0.72896992239148541981031E-001 * ramp_rate*min_down_time - 0.19118672209593035038466 * ramp_rate*marg_cst - 0.11668429237924160535300 * ramp_rate*no_load_cst - 0.44585017496470064068959 * ramp_rate*st_time_hot + 0.21804616308068380803320 * ramp_rate*st_time_warm - 0.11189531179396965121509 * ramp_rate*st_time_cold + 0.31514794333790913105631 * ramp_rate*st_cst_cold - 0.12536303571592358890863E-001 * min_up_time*marg_cst - 0.30736324939495063973682E-001 * min_up_time*st_time_hot + 0.24593719528433651000388E-001 * min_up_time*st_time_warm - 0.15059318008747154399307E-001 * min_up_time*st_cst_hot - 0.86635507482276774560148E-001 * min_down_time*st_time_hot + 0.35671748424100521712710E-001 * min_down_time*st_time_warm + 0.76714155729209601575214E-001 * min_down_time*st_cst_hot - 0.18631579077963339452495E-001 * marg_cst*no_load_cst + 0.65842146660263861646101E-001 * marg_cst*st_time_hot - 0.52599499385266507145431E-001 * marg_cst*st_time_warm + 0.45385563748822882468303E-001 * no_load_cst*st_time_hot - 0.40222464037691633975680E-001 * no_load_cst*st_time_warm + 0.22685514445037826408713E-001 * (pmax*pmin)**2 + 0.38144831211868436560142E-001 * (pmax*ramp_rate)**2 - 0.55719887726902110614002E-001 * (pmax*st_time_hot)**2 + 0.12313710638758830831496E-001 * (pmax*st_time_warm)**2 - 0.32389058155033734198724E-001 * (pmax*st_cst_hot)**2 - 0.30206148268928516981147E-001 * (pmin*ramp_rate)**2 - 0.90280446733125921388030E-002 * (pmin*marg_cst)**2 + 0.29133245121134663385698E-002 * (pmin*st_time_hot)**2 - 0.81772771309499021202827E-002 * (ramp_rate*min_up_time)**2 - 0.71613013750940479607010E-002 * (ramp_rate*min_down_time)**2 + 0.47160471362183596297224E-002 * (ramp_rate*marg_cst)**2 + 0.26488629965556051426567E-001 * (ramp_rate*st_time_hot)**2 - 0.60968042920466161693760E-001 * (ramp_rate*st_time_warm)**2 - 0.43375747262353867017559E-001 * (ramp_rate*st_time_cold)**2 + 0.10544887338202461879444 * (ramp_rate*st_cst_hot)**2 + 0.10255129535368888091251E-001 * (min_up_time*st_time_hot)**2 + 0.40911089582582056689564E-001 * (min_down_time*st_time_hot)**2 - 0.13217439596341824109560E-001 * (no_load_cst*st_cst_hot)**2

    z_unscale = z*zstd3 + zm3
    return z_unscale

with open("alamo_surrogates/hours_zone_4.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm4 = model_data["xm"]
xstd4 = model_data["xstd"]
zm4 = model_data["zm"]
zstd4 = model_data["zstd"]

def hours_zone_4(m):
    pmax = (m.pmax - xm4[0])/xstd4[0]
    pmin = (m.pmin - xm4[1])/xstd4[1]
    ramp_rate = (m.ramp_rate - xm4[2])/xstd4[2]
    min_up_time = (m.min_up_time - xm4[3])/xstd4[3]
    min_down_time = (m.min_down_time  - xm4[4])/xstd4[4]
    marg_cst = (m.marg_cst - xm4[5])/xstd4[5]
    no_load_cst = (m.no_load_cst - xm4[6])/xstd4[6]
    st_time_hot = (m.st_time_hot - xm4[7])/xstd4[7]
    st_time_warm = (m.st_time_warm - xm4[8])/xstd4[8]
    st_time_cold = (m.st_time_cold - xm4[9])/xstd4[9]
    st_cst_hot = (m.st_cst_hot - xm4[10])/xstd4[10]
    st_cst_warm = (m.st_cst_warm - xm4[11])/xstd4[11]
    st_cst_cold = (m.st_cst_cold - xm4[12])/xstd4[12]

    z = 0.53932701937149918336445 * pmax - 0.29108971984877796135294 * pmin + 0.87225053264359427018793E-002 * ramp_rate - 0.11722890924910245313839E-001 * min_up_time - 0.57228447327522524723520E-001 * min_down_time + 0.15575392281011457562556 * marg_cst + 0.47143592023564565018923E-001 * no_load_cst + 0.17964803025784978340873 * st_time_hot - 0.30210306413550311321892E-001 * st_time_warm + 0.11790910809223682786939 * st_time_cold - 0.15707551389671073338583E-001 * st_cst_hot - 0.12563728434059026617398 * st_cst_warm - 0.53465483979609751408191 * pmax**2 - 0.13128016862605568215727 * pmin**2 - 0.72476967284630366972209 * ramp_rate**2 - 0.27300323552887344885631E-002 * min_up_time**2 - 0.25235988549928332902450E-001 * min_down_time**2 + 0.83524961965659588747357E-001 * marg_cst**2 - 0.91904998588352611321728E-002 * no_load_cst**2 + 0.25448035379992012261496 * st_time_hot**2 + 0.57169167371782148023840 * pmax*pmin + 1.2545120863399548127859 * pmax*ramp_rate + 0.34297650162080375868534E-001 * pmax*min_up_time + 0.18544475147431679212806E-001 * pmax*marg_cst - 0.18310552976487015630624 * pmax*st_time_hot + 0.12113811145965573679018 * pmax*st_time_warm - 0.78486722665162378387294 * pmin*ramp_rate - 0.31499873151876037202790E-001 * pmin*min_up_time + 0.17026943855510132119768 * pmin*st_time_hot - 0.11530005963135435409495 * pmin*st_time_warm - 0.25531215198087008066974E-001 * ramp_rate*min_up_time + 0.19672158802972669611187 * ramp_rate*st_time_hot - 0.11969849794193808401044 * ramp_rate*st_time_warm + 0.82362468951413230627834E-002 * ramp_rate*st_time_cold - 0.23830141218418733617401E-001 * ramp_rate*st_cst_hot - 0.16534960102672136006419E-001 * min_up_time*marg_cst - 0.17104899953607709955916E-001 * min_up_time*no_load_cst + 0.77868636205735988764509E-001 * min_up_time*st_time_hot - 0.12993635316048737782246 * min_up_time*st_time_warm + 0.11694480235557928871071 * min_up_time*st_cst_warm - 0.18322092003957058531372E-001 * min_down_time*marg_cst - 0.15694173579602208046246 * min_down_time*st_time_hot + 0.11668176347232710188262 * min_down_time*st_time_warm + 0.26384935474372556846179E-001 * marg_cst*no_load_cst + 0.56275337435156412568826E-001 * marg_cst*st_time_hot + 0.25334095961358351484227E-001 * marg_cst*st_time_warm - 0.11383322498994076499290 * marg_cst*st_cst_warm + 0.10348531862413297421899 * no_load_cst*st_time_hot - 0.82747857252856646392836E-001 * no_load_cst*st_time_warm + 0.35914650878694727442753E-001 * (pmax*st_time_hot)**2 - 0.95024786002363068049270E-002 * (pmin*marg_cst)**2 - 0.27195745002529640027555E-001 * (pmin*st_time_hot)**2 - 0.21853216799800855713043E-001 * (pmin*st_cst_hot)**2 + 0.55920889314204835426891E-002 * (ramp_rate*min_down_time)**2 - 0.92641756876872746739870E-002 * (ramp_rate*marg_cst)**2 - 0.23030675600021508625526E-001 * (ramp_rate*st_time_hot)**2 + 0.39265555221856332512864E-001 * (ramp_rate*st_time_warm)**2 + 0.29253627608771627655049E-001 * (ramp_rate*st_time_cold)**2 - 0.72567326654445041267394E-001 * (ramp_rate*st_cst_hot)**2 + 0.25806534463545711260979E-001 * (min_down_time*st_time_hot)**2 - 0.29576646745551358336224E-001 * (marg_cst*st_time_hot)**2 - 0.18623672220170615215773E-001 * (marg_cst*st_cst_cold)**2

    z_unscale = z*zstd4 + zm4
    return z_unscale

with open("alamo_surrogates/hours_zone_5.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm5 = model_data["xm"]
xstd5 = model_data["xstd"]
zm5 = model_data["zm"]
zstd5 = model_data["zstd"]

def hours_zone_5(m):
    pmax = (m.pmax - xm5[0])/xstd5[0]
    pmin = (m.pmin - xm5[1])/xstd5[1]
    ramp_rate = (m.ramp_rate - xm5[2])/xstd5[2]
    min_up_time = (m.min_up_time - xm5[3])/xstd5[3]
    min_down_time = (m.min_down_time  - xm5[4])/xstd5[4]
    marg_cst = (m.marg_cst - xm5[5])/xstd5[5]
    no_load_cst = (m.no_load_cst - xm5[6])/xstd5[6]
    st_time_hot = (m.st_time_hot - xm5[7])/xstd5[7]
    st_time_warm = (m.st_time_warm - xm5[8])/xstd5[8]
    st_time_cold = (m.st_time_cold - xm5[9])/xstd5[9]
    st_cst_hot = (m.st_cst_hot - xm5[10])/xstd5[10]
    st_cst_warm = (m.st_cst_warm - xm5[11])/xstd5[11]
    st_cst_cold = (m.st_cst_cold - xm5[12])/xstd5[12]

    z = 0.88061901344827542281735 * pmax - 0.56366029424750108134390 * pmin - 1.4615636239468736690128 * ramp_rate + 0.67097422857155687020425E-002 * min_up_time + 0.83265121444820802687481E-001 * marg_cst - 0.96767303024194832594684E-002 * no_load_cst + 0.27139056846710812864742 * st_time_hot - 0.12161717587383390204447 * st_time_warm - 0.97593668957726600887703E-001 * st_time_cold - 0.20387078695426322227924 * st_cst_hot + 0.13577840721559553127662 * pmax**2 + 0.19820407430751235677846 * pmin**2 + 1.3291405587919613573433 * ramp_rate**2 - 0.75493721327451798058794E-002 * min_up_time**2 + 0.51645353389857852344225E-001 * marg_cst**2 + 0.52287898401867840408874E-002 * no_load_cst**2 - 0.53980859184931517802397 * st_time_hot**2 - 0.32957720613192054148755 * pmax*pmin - 1.1630748451286903044632 * pmax*ramp_rate + 0.33440330698224295102872E-001 * pmax*min_up_time + 0.43882483928852064614112E-001 * pmax*min_down_time - 0.31605173351734475173380E-001 * pmax*marg_cst - 0.10467019918504957831651 * pmax*no_load_cst - 0.32896743960266411344051 * pmax*st_time_hot + 0.17298902077138217370234 * pmax*st_time_warm - 0.70080048563294403130008E-001 * pmax*st_time_cold + 0.13291484753121424189359 * pmax*st_cst_hot + 0.87498999613996841784314 * pmin*ramp_rate - 0.18879392803491351626732E-001 * pmin*min_up_time - 0.28436935307403540112992E-001 * pmin*min_down_time - 0.61603496821709662761846E-002 * pmin*marg_cst + 0.62472037068269967163836E-001 * pmin*no_load_cst + 0.18319908316282937366104 * pmin*st_time_hot - 0.93623138258127819311127E-001 * pmin*st_time_warm + 0.40701479552985690701927E-001 * pmin*st_time_cold - 0.85772508138720199299954E-001 * pmin*st_cst_hot - 0.44754462828432227394782E-001 * ramp_rate*min_up_time - 0.56803342812735517497469E-001 * ramp_rate*min_down_time + 0.33258421281662273183422E-001 * ramp_rate*marg_cst + 0.13118692033669959728925 * ramp_rate*no_load_cst + 0.44377309522981644995809 * ramp_rate*st_time_hot - 0.22802614160439624302334 * ramp_rate*st_time_warm + 0.97866684852271887407049E-001 * ramp_rate*st_time_cold - 0.19869408263765880873208 * ramp_rate*st_cst_hot - 0.91379179676388932324071E-002 * marg_cst*no_load_cst - 0.13025162909327026722339E-001 * marg_cst*st_cst_hot + 0.12814045806303691818484E-001 * no_load_cst*st_time_hot - 0.17080339704843187226269E-001 * no_load_cst*st_time_warm - 0.68000306771793167515128E-001 * (pmax*ramp_rate)**2 + 0.50584952823051483605798E-001 * (pmax*st_time_hot)**2 + 0.88388719676192294139039E-002 * (pmax*st_time_cold)**2 - 0.53958696068499352460623E-001 * (pmin*ramp_rate)**2 - 0.79168589696822274509591E-002 * (pmin*marg_cst)**2 - 0.10937475492523286366153E-001 * (pmin*st_time_hot)**2 + 0.94742645297225414752207E-002 * (ramp_rate*min_up_time)**2 - 0.12427742545633925497217E-001 * (ramp_rate*marg_cst)**2 - 0.95716927979186353786512E-002 * (ramp_rate*no_load_cst)**2 - 0.20631356514346319008801E-001 * (ramp_rate*st_time_hot)**2 + 0.34835146280662337980871E-001 * (ramp_rate*st_time_warm)**2 + 0.24431765546675789091413E-001 * (ramp_rate*st_time_cold)**2 - 0.59964112558759212479043E-001 * (ramp_rate*st_cst_hot)**2 - 0.72693114375904895399505E-002 * (min_up_time*marg_cst)**2 + 0.10564299695607940257625E-001 * (marg_cst*st_cst_hot)**2

    z_unscale = z*zstd5 + zm5
    return z_unscale

with open("alamo_surrogates/hours_zone_6.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm6 = model_data["xm"]
xstd6 = model_data["xstd"]
zm6 = model_data["zm"]
zstd6 = model_data["zstd"]

def hours_zone_6(m):
    pmax = (m.pmax - xm6[0])/xstd6[0]
    pmin = (m.pmin - xm6[1])/xstd6[1]
    ramp_rate = (m.ramp_rate - xm6[2])/xstd6[2]
    min_up_time = (m.min_up_time - xm6[3])/xstd6[3]
    min_down_time = (m.min_down_time  - xm6[4])/xstd6[4]
    marg_cst = (m.marg_cst - xm6[5])/xstd6[5]
    no_load_cst = (m.no_load_cst - xm6[6])/xstd6[6]
    st_time_hot = (m.st_time_hot - xm6[7])/xstd6[7]
    st_time_warm = (m.st_time_warm - xm6[8])/xstd6[8]
    st_time_cold = (m.st_time_cold - xm6[9])/xstd6[9]
    st_cst_hot = (m.st_cst_hot - xm6[10])/xstd6[10]
    st_cst_warm = (m.st_cst_warm - xm6[11])/xstd6[11]
    st_cst_cold = (m.st_cst_cold - xm6[12])/xstd6[12]

    z = 1.0998121833830452054315 * pmax - 0.72547739820429535395618 * pmin - 0.33594475885416941185468 * ramp_rate + 0.28016867674290610312759E-001 * min_up_time + 0.65522931026162301026972E-001 * min_down_time - 0.11268562753294046607788 * marg_cst - 0.12643372620882145640486 * no_load_cst - 0.18470000819539150738180 * st_time_hot + 0.33029414688653696563492E-001 * st_time_warm - 0.70563846806637839370602E-001 * st_time_cold + 0.19338736095211798982074E-001 * st_cst_hot + 0.12426280533446340537118 * st_cst_warm + 0.12275392864876961962484 * pmax**2 + 0.79636818026931205372065E-001 * pmin**2 + 0.16519189691667554575893 * ramp_rate**2 - 0.21878768558431793689323E-002 * min_up_time**2 + 0.20909714176526199291839E-001 * min_down_time**2 + 0.41274446207422849086655E-001 * marg_cst**2 - 0.25557588511639577066870E-001 * no_load_cst**2 - 0.13940557083917437530829 * st_time_hot**2 - 0.12692671555400485139398E-001 * pmax*pmin - 0.31919056253168437420342 * pmax*ramp_rate - 0.12869937852095355776405E-001 * pmax*min_up_time + 0.67459295309432449998244E-003 * pmax*marg_cst - 0.48880333760646620111512E-001 * pmax*no_load_cst + 0.46189717870684675327109E-001 * pmax*st_time_hot - 0.38456797225708980181302E-001 * pmax*st_time_warm + 0.16503741747408232076744 * pmin*ramp_rate + 0.14734428664189603425116E-001 * pmin*min_up_time - 0.86244658026817157953081E-002 * pmin*min_down_time - 0.89186193485932763191926E-001 * pmin*marg_cst + 0.30339688107841666225850E-001 * pmin*no_load_cst - 0.54242191786159674182777E-001 * pmin*st_time_hot + 0.41573018839664528656375E-001 * pmin*st_time_warm + 0.25761067607988933844676E-002 * pmin*st_time_cold + 0.27662319712153604633897E-001 * ramp_rate*min_up_time + 0.88440644370218826630925E-002 * ramp_rate*marg_cst - 0.68000745151954949951900E-001 * ramp_rate*st_time_hot - 0.79285364601106769061323E-002 * ramp_rate*st_time_warm + 0.86589193277224821287952E-001 * ramp_rate*st_cst_warm + 0.28527284879390907290642E-001 * min_up_time*marg_cst + 0.22519857021578968542252E-001 * min_up_time*no_load_cst + 0.43350094421951201884013E-001 * min_up_time*st_time_hot - 0.27415580979408868400604E-001 * min_up_time*st_time_warm + 0.15852613909454322388415E-001 * min_up_time*st_time_cold - 0.21566481701625807992917E-001 * min_up_time*st_cst_hot + 0.10240877146417470822115E-001 * min_down_time*no_load_cst + 0.93510949068431034603144E-001 * min_down_time*st_time_hot - 0.92657557514777381946214E-002 * min_down_time*st_time_warm + 0.39847254637600942883680E-001 * min_down_time*st_time_cold - 0.93936607361299981344693E-001 * min_down_time*st_cst_hot - 0.12086281252947500552963 * marg_cst*no_load_cst - 0.78790519659018370957071E-001 * marg_cst*st_time_hot + 0.18417030416613634682260E-002 * marg_cst*st_time_warm - 0.76281165728812527423996E-002 * marg_cst*st_time_cold + 0.77477063169015855659261E-001 * marg_cst*st_cst_warm - 0.87820703828795254608508E-001 * no_load_cst*st_time_hot + 0.26812918551374010506239E-001 * no_load_cst*st_time_warm - 0.26747633314255232772627E-001 * no_load_cst*st_time_cold + 0.64016752946681854119504E-001 * no_load_cst*st_cst_hot - 0.59356449755462441231213E-001 * (pmax*pmin)**2 + 0.28525917793581899439825E-001 * (pmax*ramp_rate)**2 - 0.19321959584268553339337E-001 * (pmax*marg_cst)**2 - 0.67974043351838936075082E-002 * (pmax*st_time_hot)**2 + 0.11847730950800191077055E-001 * (pmin*ramp_rate)**2 + 0.18955572147452439540904E-001 * (pmin*marg_cst)**2 - 0.29996802832059029676748E-002 * (pmin*st_time_hot)**2 - 0.56271935033951614446579E-002 * (pmin*st_time_warm)**2 - 0.63532111690007371171407E-002 * (pmin*st_time_cold)**2 + 0.25872643416271191851852E-001 * (pmin*st_cst_hot)**2 - 0.15019189226513190646428E-001 * (ramp_rate*marg_cst)**2 + 0.61580738724968759820189E-002 * (ramp_rate*no_load_cst)**2 + 0.13124157942608209378998E-001 * (ramp_rate*st_time_hot)**2 + 0.79130022788374432957115E-002 * (ramp_rate*st_cst_hot)**2 + 0.59190820515248468783320E-002 * (min_up_time*marg_cst)**2 - 0.21346599389980877407857E-002 * (min_up_time*st_time_hot)**2 - 0.39198097506592775862710E-002 * (min_up_time*st_time_cold)**2 + 0.22256581912372063423999E-002 * (min_up_time*st_cst_hot)**2 - 0.52457436084188445793441E-002 * (min_down_time*marg_cst)**2 - 0.38518251053209226833496E-002 * (min_down_time*no_load_cst)**2 - 0.11395058271581763598146E-001 * (min_down_time*st_time_hot)**2 + 0.39967323649217909378728E-002 * (min_down_time*st_time_warm)**2 + 0.25277551253204215940540E-002 * (min_down_time*st_time_cold)**2 - 0.23846046823253825436284E-001 * (min_down_time*st_cst_hot)**2 - 0.23247554458443408370583E-001 * (marg_cst*no_load_cst)**2 - 0.21640812428943952128169E-001 * (marg_cst*st_time_hot)**2 - 0.82787388243743785520845E-002 * (marg_cst*st_time_warm)**2 - 0.89672782274752161485454E-002 * (marg_cst*st_time_cold)**2 + 0.34228103772491233480757E-001 * (marg_cst*st_cst_hot)**2 + 0.53431446095793554140752E-001 * (no_load_cst*st_time_hot)**2 - 0.13274305474924757639044E-001 * (no_load_cst*st_time_warm)**2

    z_unscale = z*zstd6 + zm6
    return z_unscale

with open("alamo_surrogates/hours_zone_7.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm7 = model_data["xm"]
xstd7 = model_data["xstd"]
zm7 = model_data["zm"]
zstd7 = model_data["zstd"]

def hours_zone_7(m):
    pmax = (m.pmax - xm7[0])/xstd7[0]
    pmin = (m.pmin - xm7[1])/xstd7[1]
    ramp_rate = (m.ramp_rate - xm7[2])/xstd7[2]
    min_up_time = (m.min_up_time - xm7[3])/xstd7[3]
    min_down_time = (m.min_down_time  - xm7[4])/xstd7[4]
    marg_cst = (m.marg_cst - xm7[5])/xstd7[5]
    no_load_cst = (m.no_load_cst - xm7[6])/xstd7[6]
    st_time_hot = (m.st_time_hot - xm7[7])/xstd7[7]
    st_time_warm = (m.st_time_warm - xm7[8])/xstd7[8]
    st_time_cold = (m.st_time_cold - xm7[9])/xstd7[9]
    st_cst_hot = (m.st_cst_hot - xm7[10])/xstd7[10]
    st_cst_warm = (m.st_cst_warm - xm7[11])/xstd7[11]
    st_cst_cold = (m.st_cst_cold - xm7[12])/xstd7[12]

    z = 1.1529422400494586664621 * pmax - 0.70280044041956302169893 * pmin - 0.34742720404260729605284 * ramp_rate + 0.32798406103592317450968E-001 * min_up_time + 0.40490233188166441236078E-001 * min_down_time - 0.96024788991982926167346E-001 * marg_cst - 0.12342575475210930402437 * no_load_cst - 0.16711143023738325719130 * st_time_hot + 0.16430729233047870752049E-001 * st_time_warm - 0.47053450231488117005973E-001 * st_time_cold + 0.60052218280540817230140E-001 * st_cst_hot + 0.12560489027920568316787 * st_cst_warm + 0.60228795081499190977592E-001 * pmax**2 - 0.11643329263938229090058 * pmin**2 + 0.15874045060241870341855 * ramp_rate**2 - 0.79404477893335774496553E-002 * min_up_time**2 + 0.19357048809756752932865E-001 * min_down_time**2 - 0.19054945953650344889554E-003 * marg_cst**2 - 0.15663628841242980410664E-001 * no_load_cst**2 - 0.40585391620646449339915E-001 * st_time_hot**2 - 0.51172145108315265221721E-002 * pmax*pmin - 0.13371947282563712011694 * pmax*ramp_rate - 0.16062860484567693247770E-001 * pmax*min_up_time - 0.14453143287452493992440E-001 * pmax*marg_cst - 0.30046801143263753008927E-001 * pmax*no_load_cst + 0.82863698130145060916085E-001 * pmax*st_time_hot - 0.63993645746483238112567E-001 * pmax*st_time_warm + 0.45985225486887748233555E-002 * pmax*st_time_cold + 0.10908210475877240519527 * pmin*ramp_rate + 0.16152320415399382819155E-001 * pmin*min_up_time - 0.62732854567529878347942E-002 * pmin*min_down_time - 0.43431481524091868073878E-001 * pmin*marg_cst + 0.92298301269097497018246E-002 * pmin*no_load_cst - 0.72625399451773237080587E-001 * pmin*st_time_hot + 0.54262488552878518355271E-001 * pmin*st_time_warm + 0.25781260375668572926156E-001 * ramp_rate*min_up_time - 0.39301935699509786548145E-001 * ramp_rate*marg_cst - 0.10323029327067564930864E-001 * ramp_rate*no_load_cst - 0.90799770158086046123458E-001 * ramp_rate*st_time_hot + 0.21901183789011170049976E-001 * ramp_rate*st_time_warm - 0.32167145970040636644705E-002 * ramp_rate*st_time_cold + 0.71168387417291867347302E-001 * ramp_rate*st_cst_warm + 0.28125729121647964625641E-001 * min_up_time*marg_cst + 0.30385278706965026390696E-001 * min_up_time*no_load_cst + 0.53229762877513929486550E-001 * min_up_time*st_time_hot - 0.39675182998898772535057E-001 * min_up_time*st_time_warm + 0.79729869027415148363680E-002 * min_up_time*st_time_cold - 0.50269598499369050424029E-003 * min_up_time*st_cst_hot + 0.57687029813780058007389E-001 * min_down_time*st_time_hot + 0.68554926380256465609508E-002 * min_down_time*st_time_warm + 0.31481544524341611990259E-001 * min_down_time*st_time_cold - 0.80022555776736475907640E-001 * min_down_time*st_cst_hot - 0.12762607639453751873226 * marg_cst*no_load_cst - 0.11842390357818761981168 * marg_cst*st_time_hot + 0.57435706368672581612067E-001 * marg_cst*st_time_warm - 0.18360560971489962855951E-001 * marg_cst*st_time_cold + 0.36462311113249136484971E-001 * marg_cst*st_cst_hot - 0.85246372941035070125437E-001 * no_load_cst*st_time_hot + 0.29897075936894005471212E-001 * no_load_cst*st_time_warm - 0.24988685139334103346709E-001 * no_load_cst*st_time_cold + 0.52472887425165613828337E-001 * no_load_cst*st_cst_hot + 0.39286553080670826021414E-001 * (pmax*pmin)**2 - 0.31126844835378329745534E-001 * (pmax*ramp_rate)**2 + 0.42644352454707197194739E-002 * (pmax*marg_cst)**2 - 0.90307956977165055462153E-002 * (pmax*no_load_cst)**2 - 0.12879177840163963039699E-001 * (pmax*st_time_hot)**2 + 0.66714151237682482384339E-002 * (pmax*st_cst_hot)**2 + 0.21167860252382229846457E-001 * (pmin*st_time_hot)**2 + 0.85964272599151471485751E-003 * (pmin*st_time_cold)**2 + 0.11617448483290068392271E-001 * (pmin*st_cst_hot)**2 + 0.87736736267485463530713E-002 * (ramp_rate*marg_cst)**2 + 0.65612630302162638570862E-002 * (ramp_rate*no_load_cst)**2 + 0.18482517501285069400074E-001 * (ramp_rate*st_time_hot)**2 + 0.42065898663438258328617E-002 * (min_up_time*st_time_hot)**2 - 0.34510585412433469193472E-003 * (min_up_time*st_cst_hot)**2 - 0.61113411565519642695832E-002 * (min_down_time*marg_cst)**2 - 0.90169363306052147805092E-002 * (min_down_time*st_time_hot)**2 - 0.15847523625197267732601E-001 * (min_down_time*st_cst_hot)**2 - 0.27189873575845912440085E-001 * (marg_cst*no_load_cst)**2 + 0.92387549928170994290033E-002 * (marg_cst*st_time_hot)**2 - 0.69613970030276807812730E-002 * (marg_cst*st_time_warm)**2 - 0.50606268617803869952487E-003 * (marg_cst*st_time_cold)**2 + 0.15833623694887968141876E-001 * (marg_cst*st_cst_hot)**2 + 0.40862673148763073438516E-001 * (no_load_cst*st_time_hot)**2 - 0.10138508310980502702403E-001 * (no_load_cst*st_time_warm)**2
    0.53431446095793554140752E-001 * (no_load_cst*st_time_hot)**2 - 0.13274305474924757639044E-001 * (no_load_cst*st_time_warm)**2

    z_unscale = z*zstd7 + zm7
    return z_unscale

with open("alamo_surrogates/hours_zone_8.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm8 = model_data["xm"]
xstd8 = model_data["xstd"]
zm8 = model_data["zm"]
zstd8 = model_data["zstd"]

def hours_zone_8(m):
    pmax = (m.pmax - xm8[0])/xstd8[0]
    pmin = (m.pmin - xm8[1])/xstd8[1]
    ramp_rate = (m.ramp_rate - xm8[2])/xstd8[2]
    min_up_time = (m.min_up_time - xm8[3])/xstd8[3]
    min_down_time = (m.min_down_time  - xm8[4])/xstd8[4]
    marg_cst = (m.marg_cst - xm8[5])/xstd8[5]
    no_load_cst = (m.no_load_cst - xm8[6])/xstd8[6]
    st_time_hot = (m.st_time_hot - xm8[7])/xstd8[7]
    st_time_warm = (m.st_time_warm - xm8[8])/xstd8[8]
    st_time_cold = (m.st_time_cold - xm8[9])/xstd8[9]
    st_cst_hot = (m.st_cst_hot - xm8[10])/xstd8[10]
    st_cst_warm = (m.st_cst_warm - xm8[11])/xstd8[11]
    st_cst_cold = (m.st_cst_cold - xm8[12])/xstd8[12]

    z = 0.78896059742890412014305 * pmax - 0.47815260127782893695425 * pmin - 0.50946039670550458122733 * ramp_rate + 0.49434524648980706501566E-002 * min_up_time - 0.64404671427732720911941E-001 * min_down_time + 0.10732700489923964470851 * marg_cst + 0.12577621121147349730895E-001 * st_time_hot - 0.73895641228884423323819E-001 * st_time_warm + 0.64446527562318584037016E-001 * st_time_cold + 0.10022145820929649617792 * st_cst_warm - 0.67426439173459062903504 * pmax**2 - 0.27678573453954213867689 * pmin**2 - 0.93681089503428893738857 * ramp_rate**2 + 0.51438137610549270958837E-002 * min_up_time**2 - 0.18663283785271132103611E-001 * min_down_time**2 + 0.10832479723084532940991 * marg_cst**2 - 0.19990032019274951019527E-001 * no_load_cst**2 + 0.19498222050373589797623 * st_time_hot**2 + 0.87996284618421094503304 * pmax*pmin + 1.6894498296522546532117 * pmax*ramp_rate - 0.33849268251132512419499E-001 * pmax*min_up_time - 0.77171792235463526621331E-001 * pmax*min_down_time + 0.16114991589217692036229 * pmax*marg_cst + 0.11682184766450313950781 * pmax*no_load_cst + 0.41845201674039428363372 * pmax*st_time_hot - 0.20993812449325791047450 * pmax*st_time_warm + 0.10202369160411864368321 * pmax*st_time_cold - 0.21684374059395450373700 * pmax*st_cst_hot - 1.1583472102326020092278 * pmin*ramp_rate + 0.22631506178519028571472E-001 * pmin*min_up_time + 0.33768736288019553237749E-001 * pmin*min_down_time - 0.88161276883299818929451E-001 * pmin*marg_cst - 0.73189880256093395316519E-001 * pmin*no_load_cst - 0.21358649749675637230517 * pmin*st_time_hot + 0.11114935266145709036323 * pmin*st_time_warm - 0.47722830169338294192816E-001 * pmin*st_time_cold + 0.93208328668136442018977E-001 * pmin*st_cst_hot + 0.45700656138280032481092E-001 * ramp_rate*min_up_time + 0.11245974824750633314085 * ramp_rate*min_down_time - 0.19456894861180920597299 * ramp_rate*marg_cst - 0.15124758916646072193224 * ramp_rate*no_load_cst - 0.54535589101579906579076 * ramp_rate*st_time_hot + 0.26802181778559641678328 * ramp_rate*st_time_warm - 0.13477359415979808798802 * ramp_rate*st_time_cold + 0.37165861890221218999386 * ramp_rate*st_cst_cold - 0.11605616084972136120568E-002 * min_up_time*st_time_hot - 0.50903921652123776397936E-002 * min_up_time*st_time_warm - 0.15592558165547417206587E-001 * min_down_time*marg_cst - 0.10316179917675576127589 * min_down_time*st_time_hot + 0.57832328167852158073980E-001 * min_down_time*st_time_warm + 0.55970092828270388840561E-001 * min_down_time*st_cst_hot - 0.12898127523485821213645E-001 * marg_cst*no_load_cst + 0.66360233963483006291995E-001 * marg_cst*st_time_hot - 0.51091021329693508956638E-001 * marg_cst*st_time_warm + 0.47714813441864303844575E-001 * no_load_cst*st_time_hot - 0.42744362896941917595584E-001 * no_load_cst*st_time_warm - 0.12461202155884407161146E-001 * (pmax*marg_cst)**2 - 0.41834999163172688352308E-002 * (pmax*st_time_hot)**2 + 0.42165358946598639755532E-001 * (pmax*st_time_warm)**2 + 0.31884257839868930528571E-001 * (pmax*st_time_cold)**2 - 0.70209997917373784259709E-001 * (pmax*st_cst_hot)**2 - 0.21917144370632277855515E-001 * (pmin*ramp_rate)**2 - 0.13953069898104207557932E-001 * (pmin*marg_cst)**2 + 0.49587171091729428940020E-002 * (pmin*st_cst_cold)**2 - 0.86081458291024284118498E-002 * (ramp_rate*min_up_time)**2 - 0.90265423953653934480146E-002 * (ramp_rate*min_down_time)**2 + 0.65981701216602257203947E-002 * (ramp_rate*marg_cst)**2 + 0.12741877686149727724740E-001 * (ramp_rate*no_load_cst)**2 + 0.17667328310366291199696E-001 * (ramp_rate*st_time_hot)**2 - 0.79475716411972083252913E-001 * (ramp_rate*st_time_warm)**2 - 0.59673796354557995247347E-001 * (ramp_rate*st_time_cold)**2 + 0.13650513238271261284851 * (ramp_rate*st_cst_hot)**2 - 0.13319854830947017686194E-002 * (min_up_time*st_time_hot)**2 + 0.40314408982679926507497E-001 * (min_down_time*st_time_hot)**2 - 0.29719949413447872138549E-001 * (marg_cst*st_time_hot)**2
    z_unscale = z*zstd8 + zm8
    return z_unscale

with open("alamo_surrogates/hours_zone_9.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm9 = model_data["xm"]
xstd9 = model_data["xstd"]
zm9 = model_data["zm"]
zstd9 = model_data["zstd"]

def hours_zone_9(m):
    pmax = (m.pmax - xm9[0])/xstd9[0]
    pmin = (m.pmin - xm9[1])/xstd9[1]
    ramp_rate = (m.ramp_rate - xm9[2])/xstd9[2]
    min_up_time = (m.min_up_time - xm9[3])/xstd9[3]
    min_down_time = (m.min_down_time  - xm9[4])/xstd9[4]
    marg_cst = (m.marg_cst - xm9[5])/xstd9[5]
    no_load_cst = (m.no_load_cst - xm9[6])/xstd9[6]
    st_time_hot = (m.st_time_hot - xm9[7])/xstd9[7]
    st_time_warm = (m.st_time_warm - xm9[8])/xstd9[8]
    st_time_cold = (m.st_time_cold - xm9[9])/xstd9[9]
    st_cst_hot = (m.st_cst_hot - xm9[10])/xstd9[10]
    st_cst_warm = (m.st_cst_warm - xm9[11])/xstd9[11]
    st_cst_cold = (m.st_cst_cold - xm9[12])/xstd9[12]

    z = 1.1081228214355816064085 * pmax - 0.71042709116406732583471 * pmin - 0.26244730692955298145819 * ramp_rate + 0.24921777083595417906503E-001 * min_up_time + 0.40592190638834971250226E-001 * min_down_time - 0.15332457031267060498791 * marg_cst - 0.12361428819219352770453 * no_load_cst - 0.22942798729410945890450 * st_time_hot + 0.25169063604604816064558E-001 * st_time_warm - 0.27243778149248394637727E-001 * st_time_cold + 0.70071596460003798823024E-001 * st_cst_hot + 0.16093692208717932934370 * st_cst_warm - 0.25158479530305455362793 * pmax**2 - 0.77512843808492049024750E-001 * pmin**2 - 0.12477581265581677572030 * ramp_rate**2 + 0.46081464581555469511853E-002 * min_up_time**2 + 0.40919598217442627874352E-001 * min_down_time**2 + 0.10333098103875561213361 * marg_cst**2 - 0.25393315416196063361021E-001 * no_load_cst**2 + 0.10989160932618520505333 * st_time_hot**2 + 0.10353218708311616447215 * pmax*pmin + 0.24849757174509301549392 * pmax*ramp_rate + 0.66880368449184898413384E-002 * pmax*min_up_time + 0.20774796649846476864765E-001 * pmax*marg_cst - 0.31515917494402613530102E-001 * pmax*no_load_cst - 0.13430127307086911492284E-001 * pmax*st_time_hot - 0.51448357374328235191996E-001 * pmax*st_time_warm + 0.83095127541243818392047E-001 * pmax*st_cst_warm - 0.11173049607360696633407 * pmin*ramp_rate - 0.18286689569763581869610E-001 * pmin*marg_cst + 0.89461292642546361153499E-002 * pmin*no_load_cst + 0.39333644772488506394237E-002 * pmin*st_time_hot + 0.37152484538445309548982E-002 * pmin*st_time_warm + 0.52477603681842007604663E-003 * pmin*st_time_cold - 0.10638050605804002038401E-001 * ramp_rate*min_down_time + 0.19554184288537300451249E-001 * ramp_rate*marg_cst + 0.34062520537113921692551E-001 * ramp_rate*st_time_hot + 0.77615205519829394420483E-002 * ramp_rate*st_time_cold - 0.53630473947197582207380E-001 * ramp_rate*st_cst_hot + 0.30215400361594348110916E-001 * min_up_time*marg_cst + 0.25360413551704371476481E-001 * min_up_time*no_load_cst + 0.52051827169859368055205E-002 * min_up_time*st_time_hot + 0.10427799934920295443774E-001 * min_up_time*st_time_warm + 0.33847849275746411112920E-001 * min_up_time*st_time_cold - 0.53730370854552053638820E-001 * min_up_time*st_cst_hot + 0.85986915835019727544219E-002 * min_down_time*no_load_cst + 0.84087541149727579314899E-001 * min_down_time*st_time_hot - 0.26474289051017039359204E-001 * min_down_time*st_time_warm - 0.33788636163165665371455E-001 * min_down_time*st_cst_hot - 0.14320830002451079288051 * marg_cst*no_load_cst - 0.81857692014692146065258E-001 * marg_cst*st_time_hot - 0.36989838950267444304953E-002 * marg_cst*st_time_warm - 0.67462271620799619045727E-002 * marg_cst*st_time_cold + 0.91080206770267665983276E-001 * marg_cst*st_cst_warm - 0.66360073469894351183562E-001 * no_load_cst*st_time_hot + 0.67020141659522589408504E-002 * no_load_cst*st_time_warm - 0.27336447737889218306817E-001 * no_load_cst*st_time_cold + 0.68713183918676271066950E-001 * no_load_cst*st_cst_hot + 0.38754518027330646379180E-001 * (pmax*pmin)**2 + 0.14932485623179269154659E-001 * (pmax*ramp_rate)**2 - 0.13202790710467246504400E-001 * (pmax*marg_cst)**2 - 0.87245201004190042065600E-002 * (pmin*marg_cst)**2 + 0.10017049154628801608397E-002 * (pmin*st_time_hot)**2 - 0.64290911177954527047640E-002 * (pmin*st_time_warm)**2 - 0.51477912625935533241783E-002 * (pmin*st_time_cold)**2 + 0.23808086526116791631358E-001 * (pmin*st_cst_hot)**2 + 0.10142808391130190420748E-001 * (ramp_rate*marg_cst)**2 + 0.65216688272407693713340E-002 * (ramp_rate*no_load_cst)**2 - 0.14608605962872653338813E-001 * (ramp_rate*st_time_hot)**2 - 0.73908381887034836632555E-002 * (min_up_time*min_down_time)**2 + 0.36641781732958903644581E-002 * (min_up_time*marg_cst)**2 - 0.64140065326939192311140E-002 * (min_up_time*st_time_cold)**2 - 0.75825609872492982788117E-002 * (min_down_time*marg_cst)**2 - 0.12183641997380043400789E-001 * (min_down_time*st_time_hot)**2 - 0.75003933490629106847769E-002 * (min_down_time*st_cst_warm)**2 - 0.29224201207981446298811E-001 * (marg_cst*no_load_cst)**2 - 0.46406340137677012935846E-001 * (marg_cst*st_time_hot)**2 - 0.10789930120848563455116E-001 * (marg_cst*st_time_warm)**2 - 0.12599498946926385167799E-001 * (marg_cst*st_time_cold)**2 + 0.52016598210673968549766E-001 * (marg_cst*st_cst_hot)**2 + 0.37651400238560492339523E-001 * (no_load_cst*st_time_hot)**2 - 0.99306085773719580261920E-002 * (no_load_cst*st_time_warm)**2

    z_unscale = z*zstd9 + zm9
    return z_unscale

with open("alamo_surrogates/hours_zone_10.pkl", "rb") as input_file:
    model_data = pickle.load(input_file)
xm10 = model_data["xm"]
xstd10 = model_data["xstd"]
zm10 = model_data["zm"]
zstd10 = model_data["zstd"]

def hours_zone_10(m):
    pmax = (m.pmax - xm10[0])/xstd10[0]
    pmin = (m.pmin - xm10[1])/xstd10[1]
    ramp_rate = (m.ramp_rate - xm10[2])/xstd10[2]
    min_up_time = (m.min_up_time - xm10[3])/xstd10[3]
    min_down_time = (m.min_down_time  - xm10[4])/xstd10[4]
    marg_cst = (m.marg_cst - xm10[5])/xstd10[5]
    no_load_cst = (m.no_load_cst - xm10[6])/xstd10[6]
    st_time_hot = (m.st_time_hot - xm10[7])/xstd10[7]
    st_time_warm = (m.st_time_warm - xm10[8])/xstd10[8]
    st_time_cold = (m.st_time_cold - xm10[9])/xstd10[9]
    st_cst_hot = (m.st_cst_hot - xm10[10])/xstd10[10]
    st_cst_warm = (m.st_cst_warm - xm10[11])/xstd10[11]
    st_cst_cold = (m.st_cst_cold - xm10[12])/xstd10[12]

    z = -0.26236885646670859983942 * pmax + 0.32117723436356555855031E-001 * pmin + 0.18357751866704333232327 * ramp_rate + 0.41357009364525691014203E-002 * min_up_time + 0.77697696372047266380467E-002 * min_down_time - 0.63311194614721699736037 * marg_cst - 0.40609479472620120532600E-001 * no_load_cst - 0.28785159808369414902174 * st_time_hot + 0.85705676833423158345049E-001 * st_time_warm + 0.60689471393941485377610E-001 * st_time_cold + 0.28912565040310894692865 * st_cst_hot + 0.24070561865434800946639E-001 * pmax**2 + 0.21550822251072302565555E-001 * pmin**2 + 0.14661372593453261131691E-001 * ramp_rate**2 - 0.10193550684609265985503E-001 * min_up_time**2 - 0.30204794258629750515477E-002 * min_down_time**2 - 0.45699452556096198385660 * marg_cst**2 - 0.63082924170491286308682E-001 * no_load_cst**2 + 0.55056773043846030102344 * st_time_hot**2 - 0.64628754795654316800402E-001 * pmax*pmin - 0.96271060106063441330626E-001 * pmax*ramp_rate - 0.59140630982169407892091E-001 * pmax*marg_cst + 0.54527591846714332582402E-002 * pmax*no_load_cst - 0.17719443497730470206408E-002 * pmax*st_time_hot - 0.12012273515337006102310E-002 * pmax*st_time_warm + 0.27804359396462713198417E-002 * pmax*st_time_cold + 0.58213125907898830013742E-001 * pmin*ramp_rate - 0.36212507711347822342285E-002 * pmin*marg_cst - 0.12820613013326665960423E-001 * pmin*no_load_cst - 0.74273097747218803921232E-002 * pmin*st_time_hot - 0.43517072755103018699696E-002 * pmin*st_time_warm + 0.14322017093063341142134E-001 * pmin*st_cst_warm - 0.45793255866717263655175E-002 * ramp_rate*min_up_time + 0.28638818670744348865442E-001 * ramp_rate*marg_cst + 0.27407104968466523831072E-001 * ramp_rate*st_time_hot - 0.13102603617448659034661E-001 * ramp_rate*st_time_warm - 0.91136912986285126964114E-002 * ramp_rate*st_cst_hot + 0.93882783294243522809186E-002 * min_up_time*marg_cst + 0.86383530130608315866780E-002 * min_up_time*no_load_cst + 0.61372954005672634633650E-002 * min_up_time*st_time_hot - 0.40970065432930765844666E-002 * min_up_time*st_time_warm + 0.38886951966198455774015E-002 * min_up_time*st_time_cold - 0.70817202684968641804297E-002 * min_down_time*marg_cst + 0.74335984472641931583570E-002 * min_down_time*st_time_hot + 0.35487520351513280984779E-002 * min_down_time*st_time_warm - 0.45139071161162190592986E-001 * marg_cst*no_load_cst - 0.41317929393197394549730E-001 * marg_cst*st_time_hot + 0.17764391730309629646722E-001 * marg_cst*st_time_warm - 0.31276150416076928077735E-002 * marg_cst*st_time_cold + 0.21837190119048388581291E-001 * marg_cst*st_cst_hot - 0.16368783576796462619907E-001 * no_load_cst*st_time_hot - 0.14299123844040254743826E-001 * no_load_cst*st_time_warm - 0.12938803954410191959790E-001 * no_load_cst*st_time_cold + 0.32074776849295889846747E-001 * no_load_cst*st_cst_hot + 0.56631897782241187241925E-002 * (pmax*ramp_rate)**2 + 0.21206082492141994921830E-001 * (pmax*marg_cst)**2 + 0.10039892997962302986781E-001 * (pmax*no_load_cst)**2 + 0.64771337746911553448492E-002 * (pmax*st_cst_hot)**2 + 0.71576758708197180461341E-002 * (pmin*ramp_rate)**2 - 0.10171174356435287375322E-001 * (pmin*marg_cst)**2 + 0.80848820678597541067312E-002 * (pmin*st_time_hot)**2 - 0.78680476820620435379761E-002 * (ramp_rate*marg_cst)**2 - 0.68898535337331406197547E-002 * (ramp_rate*st_time_hot)**2 - 0.63360134080739198558785E-002 * (ramp_rate*st_cst_hot)**2 + 0.50390340335870510701799E-002 * (min_up_time*marg_cst)**2 + 0.54897479162764519639017E-002 * (min_up_time*st_time_hot)**2 - 0.62736272467635457311674E-002 * (min_down_time*marg_cst)**2 + 0.68819784862661241217618E-002 * (min_down_time*st_time_hot)**2 - 0.22876195692210759408125E-002 * (min_down_time*st_time_warm)**2 - 0.20925491631246150026069E-001 * (marg_cst*no_load_cst)**2 - 0.73194427691747804276545E-001 * (marg_cst*st_time_hot)**2 - 0.13192591520814483593838E-001 * (marg_cst*st_time_warm)**2 - 0.19262923857510938496773E-001 * (marg_cst*st_time_cold)**2 + 0.66106655767125183098543E-001 * (marg_cst*st_cst_hot)**2 + 0.51567467585579403188678E-001 * (no_load_cst*st_time_hot)**2 + 0.10389677564185904531935E-001 * (no_load_cst*st_time_warm)**2 + 0.18114396924220670104244E-001 * (no_load_cst*st_time_cold)**2 - 0.28543017087687864302703E-001 * (no_load_cst*st_cst_hot)**2


    z_unscale = z*zstd10 + zm10
    return z_unscale
