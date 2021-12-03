import pickle
from pyomo.common.fileutils import this_file_dir
#Revenue surrogate using only 5 terms
with open(this_file_dir()+"/revenue_scaled_cutoff_0_5_terms_model_2.pkl", "rb") as input_file:
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
    z = 0.54209913492104144783212 * pmax - 0.45490811487193222317771E-001 * pmin - 0.40615306725695556055999E-001 * ramp_rate - 0.69796847596385414025377 * marg_cst - 0.87156755581544093081092E-001 * no_load_cst - 0.26098063153092970062330E-001 * pmax**2 + 0.51136259487369134513202E-001 * pmin**2 + 0.42868070582146571706472E-002 * ramp_rate**2 - 0.36803296277759606214275 * marg_cst**2 - 0.26893694308215251170813E-001 * no_load_cst**2 - 0.37233893879685363292875E-002 * pmax*pmin + 0.64727809381044604589150E-002 * pmax*ramp_rate - 0.20718963778454951851771 * pmax*marg_cst - 0.43620594360541956047150E-001 * pmax*no_load_cst - 0.78760862718230215812065E-002 * pmin*ramp_rate - 0.30659784733526089517408E-001 * pmin*marg_cst + 0.19778388141193398164219E-001 * pmin*no_load_cst + 0.94953698308689916951497E-002 * ramp_rate*no_load_cst - 0.74378234703050424836412E-001 * marg_cst*no_load_cst + 0.33079974419997817394398E-003 * (pmax*pmin)**2 + 0.16929337207471845724166E-002 * (pmax*ramp_rate)**2 + 0.19566567746046457237918E-001 * (pmax*marg_cst)**2 - 0.28412799289819498996246E-003 * (pmax*no_load_cst)**2 - 0.15939260454833246386658E-003 * (pmin*ramp_rate)**2 - 0.43065476801435763343218E-001 * (pmin*marg_cst)**2 - 0.89669149854940046352053E-003 * (pmin*no_load_cst)**2 - 0.18029084445369825129291E-001 * (ramp_rate*marg_cst)**2 + 0.33355285853445570433407E-002 * (ramp_rate*no_load_cst)**2 + 0.27796094336623340670389E-001 * (marg_cst*no_load_cst)**2 + 0.37046174051608721233819 
    
    z_unscale = z*zstd_rev + zm_rev
    return z_unscale

#revenue surrogate using all terms
with open(this_file_dir()+"/revenue_scaled_cutoff_0_all_terms_model_1.pkl", "rb") as input_file:
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

    z = 0.54112963579100148425738 * pmax - 0.44598043564020389828428E-001 * pmin - 0.40704034001723336799738E-001 * ramp_rate + 0.53278526557970495858285E-002 * min_up_time + 0.11296427666174122320109E-001 * min_down_time - 0.69796847596397637580878 * marg_cst - 0.87156755581576733638016E-001 * no_load_cst - 0.14883518388934019460734 * st_time_hot + 0.53150343479103870358848E-001 * st_time_warm + 0.67657074772286732167181E-001 * st_time_cold + 0.65014963564026709286203E-001 * st_cst_hot - 0.31380396928615064688906E-002 * pmax**2 + 0.10064457144020877432666E-001 * pmin**2 - 0.34653608173081562777995E-002 * ramp_rate**2 - 0.15804082747403718888640E-002 * min_up_time**2 - 0.59471198871261145985079E-002 * min_down_time**2 - 0.38176486194174091837183 * marg_cst**2 + 0.30571091223063760929091E-002 * no_load_cst**2 + 0.38638540345537375486629 * st_time_hot**2 - 0.95542818585049402863030E-002 * pmax*pmin - 0.20210364983761175405874 * pmax*marg_cst - 0.44058382956577664402165E-001 * pmax*no_load_cst + 0.15720909679564788047346E-001 * pmax*st_time_hot - 0.15114995266517734090472E-001 * pmax*st_time_warm + 0.37822960178557059416488E-003 * pmax*st_time_cold - 0.25980956243562471769115E-001 * pmax*st_cst_hot - 0.33687442951894214226982E-001 * pmin*marg_cst + 0.19714296977593501886128E-001 * pmin*no_load_cst - 0.22819878269947257293238E-001 * pmin*st_time_hot + 0.22607115301569658677439E-001 * pmin*st_time_warm + 0.35024315969759437991438E-002 * pmin*st_time_cold - 0.32173285249373254068850E-001 * pmin*st_cst_hot - 0.60336517101911586041796E-002 * ramp_rate*marg_cst + 0.10445169155100918895185E-001 * ramp_rate*no_load_cst - 0.16291281715810495417385E-001 * ramp_rate*st_time_hot + 0.15198104886368180715950E-001 * ramp_rate*st_time_warm + 0.53468308002519986804613E-002 * min_up_time*marg_cst + 0.78607383686815704426643E-002 * min_up_time*st_time_hot - 0.17658818813542248062076E-002 * min_up_time*st_time_cold + 0.83784252074816691069348E-002 * min_down_time*st_time_cold - 0.74378234703050410958625E-001 * marg_cst*no_load_cst + 0.12270180592036916342180E-001 * marg_cst*st_time_hot - 0.49625900329210359193666E-002 * marg_cst*st_time_warm + 0.76009020275719426512628E-002 * marg_cst*st_time_cold - 0.10795919121565519804840 * marg_cst*st_cst_hot - 0.11069366111065612190423E-001 * no_load_cst*st_time_hot + 0.21208311976338170556922E-001 * no_load_cst*st_time_warm + 0.99440565002409230660474E-002 * no_load_cst*st_time_cold - 0.34167812240803695222890E-001 * no_load_cst*st_cst_hot 

    z_unscale = z*zstd_rev_all + zm_rev_all
    return z_unscale

with open(this_file_dir()+"/hours_zone_0.pkl", "rb") as input_file:
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

    z = - 0.47786377412680741683104E-001 * pmax - 0.46941385541625152422185 * pmin + 0.88765761990244787527082E-001 * ramp_rate + 0.31926698313291639041989E-001 * min_up_time + 0.34457460244656658299167E-001 * min_down_time + 0.26928972002567047594468 * marg_cst - 0.48252564743447946826738E-001 * no_load_cst - 0.19967223897713176627988 * st_time_hot + 0.74588251250276224602054E-001 * st_time_warm - 0.11594447494947250631991 * st_time_cold + 0.97431621273402733984792E-002 * st_cst_hot + 0.12784041357481196188317 * pmax**2 + 0.34016400517406031323020 * pmin**2 - 0.63226216567281431912839E-001 * ramp_rate**2 + 0.21344072928638074815311E-001 * min_up_time**2 + 0.30806514871900732288612E-001 * min_down_time**2 - 0.13893536561088906666761 * marg_cst**2 - 0.26422877010668123498593E-001 * no_load_cst**2 - 0.20204220285599719386660 * st_time_hot**2 - 0.91477251246889240698934E-001 * pmax*pmin - 0.92884257041113998942805E-001 * pmax*ramp_rate - 0.93043995615433271184624E-002 * pmax*min_up_time + 0.18746718723752148205719 * pmax*marg_cst + 0.45305929885082855956835E-001 * pmax*no_load_cst + 0.54283278444344790908405E-002 * pmax*st_time_hot + 0.54288793747037482340101E-001 * pmin*ramp_rate + 0.31441807165722146510944E-001 * pmin*min_up_time - 0.51330474789384639144885 * pmin*marg_cst - 0.10291135505827664342604 * pmin*no_load_cst - 0.68574066605623790193746E-001 * pmin*st_time_hot + 0.99209110901818785971384E-002 * pmin*st_time_warm - 0.29986078975943658564418E-001 * pmin*st_time_cold + 0.31589651097155148351536E-001 * min_up_time*marg_cst - 0.16889756458867966754100E-001 * min_up_time*st_time_hot + 0.72347234282653757886372E-001 * min_up_time*st_time_warm - 0.10256670056515990174795 * min_up_time*st_cst_warm - 0.86212326463182241409466E-002 * min_down_time*no_load_cst + 0.50199881958393428871279E-001 * min_down_time*st_time_hot - 0.33279972510203911784110E-001 * min_down_time*st_time_warm + 0.10302333369842407023720E-001 * min_down_time*st_time_cold - 0.21893308384599515642455E-001 * marg_cst*no_load_cst + 0.42991490436152170195871E-001 * marg_cst*st_time_hot - 0.55305781972903159116051E-001 * marg_cst*st_time_warm - 0.14581839246038185570198E-001 * marg_cst*st_time_cold + 0.28820160789959831104667E-001 * no_load_cst*st_time_hot + 0.13520124143414812695196E-001 * no_load_cst*st_time_cold - 0.75663549877734992410439E-001 * (pmax*pmin)**2 - 0.46842033790898782164014E-001 * (pmax*marg_cst)**2 + 0.31546651070172397612890E-001 * (pmin*ramp_rate)**2 - 0.10539284676903078186161E-001 * (pmin*min_up_time)**2 + 0.76488893317998560283932E-001 * (pmin*marg_cst)**2 + 0.19833083262989711220703E-001 * (pmin*no_load_cst)**2 - 0.71198660485662604302526E-001 * (pmin*st_time_hot)**2 - 0.19180641027987523766773E-001 * (pmin*st_time_warm)**2 - 0.27240384731140640517388E-001 * (pmin*st_time_cold)**2 + 0.82226840370966475246561E-001 * (pmin*st_cst_hot)**2 + 0.65409111561042301841162E-001 * (ramp_rate*marg_cst)**2 - 0.11316432047183359183018E-001 * (min_up_time*min_down_time)**2 + 0.27861638884367760649052E-001 * (marg_cst*st_time_hot)**2 - 0.35408991017373547816049E-001 * (marg_cst*st_time_warm)**2 - 0.24658386443949614069693E-001 * (marg_cst*st_time_cold)**2 + 0.65025740949107915800376E-001 * (marg_cst*st_cst_hot)**2 

    z_unscale = z*zstd0 + zm0
    return z_unscale

with open(this_file_dir()+"/hours_zone_1.pkl", "rb") as input_file:
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

    z = 0.53659502139455583780148 * pmax - 0.43990875478430230272053 * pmin + 0.25593878796106875328498E-001 * ramp_rate + 0.40495195305051433221077E-001 * min_up_time + 0.28117475924678465154516E-001 * min_down_time - 0.30474861274585013370952 * marg_cst - 0.27825490347354298048543 * no_load_cst - 0.41572909048062828274439 * st_time_hot + 0.15007163634582593170208 * st_time_warm - 0.55488267089758881722705E-001 * st_time_cold + 0.16782591893050460840620 * st_cst_hot - 0.71961509764625680371508E-002 * pmax**2 + 0.12253924345540780083130 * pmin**2 + 0.19894638680478870379487E-001 * ramp_rate**2 + 0.45457272358351112628849E-002 * min_up_time**2 + 0.30867306780413494848858E-001 * min_down_time**2 - 0.50202531913996373269526 * marg_cst**2 + 0.36310931036417248840564E-001 * no_load_cst**2 + 0.24471060356314641714981 * st_time_hot**2 + 0.20686376245449115268693E-001 * pmax*min_up_time + 0.10904355510669051440575E-001 * pmax*marg_cst - 0.13556705322189260831678 * pmax*no_load_cst - 0.46649933640303129292470E-002 * pmax*st_time_hot - 0.23015801178887976274900E-001 * pmax*st_time_warm - 0.24042340172204610532214E-001 * pmax*st_time_cold + 0.19543194140423793964123E-001 * pmin*ramp_rate - 0.17138342815993433398969E-001 * pmin*min_up_time - 0.16549252347708465205045 * pmin*marg_cst + 0.15319444823580588566081 * pmin*no_load_cst + 0.13425342029260441123473E-001 * pmin*st_time_hot + 0.25209405631366022665363E-001 * pmin*st_time_warm + 0.28718280430438385131264E-001 * pmin*st_time_cold - 0.49794829810117208213072E-001 * pmin*st_cst_hot - 0.77937291675475528496264E-001 * ramp_rate*marg_cst - 0.49134283775381842884755E-001 * ramp_rate*st_time_hot + 0.35825526531443989408654E-001 * ramp_rate*st_time_warm + 0.34526984925752603194926E-001 * min_up_time*marg_cst + 0.23722925371161458080183E-001 * min_up_time*no_load_cst + 0.91260435308793574904485E-001 * min_up_time*st_time_hot - 0.60894307832513060174673E-001 * min_up_time*st_time_warm + 0.16178740844009678601090E-001 * min_up_time*st_time_cold - 0.16671204461482309194809E-001 * min_down_time*marg_cst - 0.19953694764703333586198 * marg_cst*no_load_cst + 0.71298484531055111856901E-001 * marg_cst*st_time_hot - 0.49303543369101983373515E-001 * marg_cst*st_time_warm + 0.13390184298822827013709E-001 * marg_cst*st_time_cold - 0.15178375797167936722687 * marg_cst*st_cst_hot - 0.69303589935085049833496E-001 * no_load_cst*st_time_hot + 0.51736466052712759555732E-001 * no_load_cst*st_time_warm - 0.35846893616277304694950E-001 * (pmax*pmin)**2 + 0.25514104388074523160901E-001 * (pmax*marg_cst)**2 + 0.48346497023579728877229E-002 * (pmin*marg_cst)**2 - 0.12868342756127979301106E-001 * (pmin*st_time_hot)**2 - 0.36319457116188599843376E-001 * (ramp_rate*marg_cst)**2 + 0.26657144089343231402323E-001 * (ramp_rate*no_load_cst)**2 - 0.83749736420515967505462E-002 * (ramp_rate*st_time_hot)**2 - 0.11085696696841411748591E-001 * (min_up_time*min_down_time)**2 - 0.50473432694614402821420E-002 * (min_up_time*marg_cst)**2 - 0.94734998166636080751957E-002 * (min_up_time*no_load_cst)**2 + 0.13498829080132763655331E-001 * (min_up_time*st_time_hot)**2 - 0.58671923606474340323613E-002 * (min_down_time*marg_cst)**2 + 0.78708932248834240219537E-001 * (marg_cst*no_load_cst)**2 + 0.74770830408435656755373E-001 * (marg_cst*st_time_hot)**2 - 0.80721202978532163863079E-001 * (marg_cst*st_time_warm)**2 - 0.55587980758769910127892E-001 * (marg_cst*st_time_cold)**2 + 0.17022713055692634265270 * (marg_cst*st_cst_hot)**2 - 0.63619137541323866891219E-001 * (no_load_cst*st_time_hot)**2 - 0.32247004646399235472387E-001 * (no_load_cst*st_time_warm)**2 - 0.35191745167617805023497E-001 * (no_load_cst*st_time_cold)**2 + 0.72359131339189808329593E-001 * (no_load_cst*st_cst_hot)**2 

    z_unscale = z*zstd1 + zm1
    return z_unscale

with open(this_file_dir()+"/hours_zone_2.pkl", "rb") as input_file:
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

    z = 0.56898764156651826517930 * pmax - 0.45277758588177446918976 * pmin + 0.11036878461501616402463E-001 * ramp_rate + 0.27219626287570956868489E-001 * min_up_time + 0.20694054842051799736335E-001 * min_down_time - 0.35525698000744565518971 * marg_cst - 0.25721075908779494900713 * no_load_cst - 0.34336238667352947384614 * st_time_hot + 0.15244694460620392129080E-001 * st_time_warm + 0.19265546370183777441509E-002 * st_time_cold + 0.78701386667957382026195E-001 * st_cst_hot + 0.20182273026299962781493 * st_cst_warm - 0.93644886511927552530388E-001 * pmax**2 + 0.54351222378525021472484E-001 * pmin**2 - 0.90090648371141043848631E-001 * ramp_rate**2 - 0.11465349219931683655282E-001 * min_up_time**2 - 0.53380000799224647245467 * marg_cst**2 + 0.22409397732561089217063E-001 * no_load_cst**2 + 0.36958171094028391401309 * st_time_hot**2 + 0.95106071456909344052910E-001 * pmax*pmin + 0.20924618562396019028071 * pmax*ramp_rate + 0.13842231911208414257830E-001 * pmax*min_up_time - 0.31524713848288685169585E-001 * pmax*marg_cst - 0.11005612554033832040101 * pmax*no_load_cst + 0.24375749260819219027541E-001 * pmax*st_time_hot - 0.38735515425916987319876E-001 * pmax*st_time_warm - 0.18319285116469616325752E-001 * pmax*st_time_cold - 0.11062808449602913596355 * pmin*ramp_rate - 0.14836069646187430293161E-001 * pmin*min_up_time - 0.13175275024293231496131 * pmin*marg_cst + 0.12705491935107235401681 * pmin*no_load_cst - 0.48214258605613145858282E-001 * pmin*st_time_hot + 0.10917017229546534251572 * pmin*st_time_warm + 0.76631383550923668629418E-002 * pmin*st_time_cold - 0.10380724942851797532040 * pmin*st_cst_warm - 0.86586329398006159352441E-001 * ramp_rate*marg_cst - 0.57937307156349267256434E-001 * ramp_rate*st_time_hot + 0.44338851204342832890148E-001 * ramp_rate*st_time_warm + 0.24487472768150709417911E-001 * min_up_time*marg_cst + 0.19686235552342685323213E-001 * min_up_time*no_load_cst + 0.65315998645277720258129E-001 * min_up_time*st_time_hot - 0.43538367533336834458435E-001 * min_up_time*st_time_warm + 0.15735377357671966741526E-001 * min_up_time*st_time_cold - 0.12044319002314533492703E-001 * min_down_time*marg_cst + 0.10213258768105798815484E-001 * min_down_time*st_cst_hot - 0.18481311557861956895721 * marg_cst*no_load_cst + 0.55088432473436782510490E-001 * marg_cst*st_time_hot - 0.35040060283659725526295E-001 * marg_cst*st_time_warm + 0.15321397945547677102440E-001 * marg_cst*st_time_cold - 0.16680500061394087918387 * marg_cst*st_cst_hot - 0.59442235048644348482938E-001 * no_load_cst*st_time_hot + 0.45196755929710064059179E-001 * no_load_cst*st_time_warm - 0.14561124829993686569107E-001 * (pmax*pmin)**2 - 0.10989403240664571623220E-001 * (pmax*ramp_rate)**2 + 0.19556389528180104231403E-001 * (pmax*marg_cst)**2 - 0.16019942928101483375913E-001 * (pmax*no_load_cst)**2 + 0.13139391829467031783119E-001 * (pmin*ramp_rate)**2 + 0.14604322082647037961411E-001 * (pmin*marg_cst)**2 - 0.22767474850460090590731E-001 * (pmin*st_time_hot)**2 - 0.36676685754414815876512E-001 * (ramp_rate*marg_cst)**2 + 0.17705260198763363849173E-001 * (ramp_rate*no_load_cst)**2 - 0.81218330817846261238113E-002 * (ramp_rate*st_time_hot)**2 - 0.72816777644392691182240E-002 * (min_up_time*marg_cst)**2 + 0.19720357666869633289641E-001 * (min_up_time*st_time_hot)**2 - 0.76909936806009507326287E-002 * (min_down_time*marg_cst)**2 + 0.79601764593195192820652E-001 * (marg_cst*no_load_cst)**2 + 0.75701791945567281638674E-001 * (marg_cst*st_time_hot)**2 - 0.81190222293940045306115E-001 * (marg_cst*st_time_warm)**2 - 0.55045233420299218007887E-001 * (marg_cst*st_time_cold)**2 + 0.17445947898518676666413 * (marg_cst*st_cst_hot)**2 - 0.61319733304320063471682E-001 * (no_load_cst*st_time_hot)**2 - 0.26050754707900514867136E-001 * (no_load_cst*st_time_warm)**2 - 0.28722363589636364794577E-001 * (no_load_cst*st_time_cold)**2 + 0.58936977426532873025611E-001 * (no_load_cst*st_cst_hot)**2 

    z_unscale = z*zstd2 + zm2
    return z_unscale

with open(this_file_dir()+"/hours_zone_3.pkl", "rb") as input_file:
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

    z = 0.67954879815854285496357 * pmax - 0.43795306479872669891407 * pmin - 0.43132095202821152390982 * ramp_rate - 0.24954307589403850603516E-001 * min_up_time - 0.28180659580212350651118E-001 * min_down_time - 0.16778727687046343652888 * marg_cst - 0.74357904911739117204128E-001 * no_load_cst - 0.62861862147823438506933E-001 * st_time_hot - 0.49758548405572110284734E-001 * st_time_warm + 0.96701865347241774961695E-001 * st_time_cold + 0.11353408681175972128052 * st_cst_hot + 0.99525963087398999773470E-001 * st_cst_cold - 0.82782317719751175033593 * pmax**2 - 0.28681270059642666181432 * pmin**2 - 1.1394520864888173505847 * ramp_rate**2 - 0.72627931568685485594350E-002 * min_up_time**2 - 0.31999901217344715875779E-001 * min_down_time**2 - 0.47130244045701918942370 * marg_cst**2 - 0.73743234477960275911890E-003 * no_load_cst**2 + 0.59351103998445242648785 * st_time_hot**2 + 1.0429581946690358318364 * pmax*pmin + 1.9736235424010002414974 * pmax*ramp_rate - 0.51434752227707276894808E-001 * pmax*min_up_time - 0.45741851409169233255714E-001 * pmax*min_down_time + 0.10600901029348261395135 * pmax*marg_cst + 0.76163245239619717374957E-001 * pmax*no_load_cst + 0.33685872427519553129471 * pmax*st_time_hot - 0.14967555828945419693632 * pmax*st_time_warm + 0.95147376197331212654973E-001 * pmax*st_time_cold - 0.26414663946401550909115 * pmax*st_cst_hot - 1.2225826663111472036860 * pmin*ramp_rate + 0.24043474604501020863712E-001 * pmin*min_up_time + 0.18413377960561114893290E-001 * pmin*min_down_time - 0.13500060210516043324169 * pmin*marg_cst - 0.28076368155367813217449E-001 * pmin*no_load_cst - 0.15663560829151867981324 * pmin*st_time_hot + 0.85101384687153641683821E-001 * pmin*st_time_warm - 0.33561925607418635619794E-001 * pmin*st_time_cold + 0.78251688508474123717740E-001 * pmin*st_cst_hot + 0.72090185050372831976340E-001 * ramp_rate*min_up_time + 0.68030865449753397111721E-001 * ramp_rate*min_down_time - 0.14880148748743288189544 * ramp_rate*marg_cst - 0.10988358086716529249394 * ramp_rate*no_load_cst - 0.48259627808739763432655 * ramp_rate*st_time_hot + 0.21487130913062535420011 * ramp_rate*st_time_warm - 0.13770839110528051985050 * ramp_rate*st_time_cold + 0.38798098570582989497169 * ramp_rate*st_cst_hot - 0.99767863617206877652199E-002 * min_up_time*no_load_cst - 0.18576284297275431900420E-001 * min_up_time*st_time_hot + 0.20314726940907898433686E-001 * min_up_time*st_cst_hot - 0.12491959421572555039015E-001 * min_down_time*marg_cst + 0.97403376297964874841462E-001 * min_down_time*st_time_hot - 0.20847115564637624940403 * min_down_time*st_time_warm + 0.25428777399979113349460 * min_down_time*st_cst_warm - 0.82204743514581596341984E-001 * marg_cst*no_load_cst + 0.26829981547858022111752E-001 * marg_cst*st_time_hot + 0.12561302792822040541587E-001 * marg_cst*st_time_warm + 0.31129340436881834663918E-001 * marg_cst*st_time_cold - 0.22009015692235023298196 * marg_cst*st_cst_hot - 0.28365239548378284178964E-001 * no_load_cst*st_time_hot + 0.88339240662407503279496E-001 * no_load_cst*st_time_warm - 0.11080819517510788474457 * no_load_cst*st_cst_warm + 0.23937949466587924579608E-001 * (pmax*ramp_rate)**2 - 0.23797185136240600611401E-001 * (pmax*st_time_hot)**2 + 0.26892684122264597784691E-001 * (pmax*st_time_warm)**2 + 0.17455204821831413891653E-001 * (pmax*st_time_cold)**2 - 0.50569579347087320608800E-001 * (pmax*st_cst_hot)**2 - 0.28737501859269536097496E-001 * (pmin*ramp_rate)**2 - 0.73786252163683032495589E-002 * (pmin*st_time_hot)**2 - 0.12013595445949827589249E-001 * (ramp_rate*min_up_time)**2 - 0.57064463437094467557209E-002 * (ramp_rate*min_down_time)**2 + 0.91391924917687047263959E-002 * (ramp_rate*marg_cst)**2 + 0.14561770983654598668466E-001 * (ramp_rate*no_load_cst)**2 + 0.53096388684591752649133E-001 * (ramp_rate*st_time_hot)**2 - 0.53758607500529111133947E-001 * (ramp_rate*st_time_warm)**2 - 0.34191178356785506542437E-001 * (ramp_rate*st_time_cold)**2 + 0.91676959361995810637858E-001 * (ramp_rate*st_cst_hot)**2 + 0.77355326127591453322374E-002 * (min_up_time*st_time_hot)**2 + 0.15618059447654288746099E-001 * (min_up_time*st_cst_hot)**2 + 0.32055383071765489177984E-001 * (min_down_time*st_time_hot)**2 + 0.60086334595758168417023E-002 * (min_down_time*st_time_cold)**2 + 0.21988837705198945038365E-001 * (marg_cst*no_load_cst)**2 + 0.12941079349215342531032 * (marg_cst*st_time_hot)**2 + 0.19745053082957059881197E-001 * (marg_cst*st_time_warm)**2 + 0.32791209756843374412139E-001 * (marg_cst*st_time_cold)**2 - 0.32847229942260122714792E-001 * (no_load_cst*st_time_hot)**2 

    z_unscale = z*zstd3 + zm3
    return z_unscale

with open(this_file_dir()+"/hours_zone_4.pkl", "rb") as input_file:
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

    z = 0.40798414489436413132850 * pmax - 0.25654922853110817548838 * pmin + 0.78029302832784827245738E-002 * ramp_rate - 0.44450712812026949016797E-001 * min_up_time - 0.28061575566954002902698E-001 * min_down_time - 0.10575774300706708430653 * marg_cst - 0.87162104878162893761839E-002 * no_load_cst + 0.29937002026336534399320E-001 * st_time_hot + 0.35363198776755457031218E-001 * st_time_warm + 0.16220340510506034426008 * st_time_cold - 0.73914851437835882297378E-001 * st_cst_hot - 0.58121449278305634122965E-001 * st_cst_warm - 0.71253540751554406140755E-001 * st_cst_cold - 0.60211384709430781914108 * pmax**2 - 0.66989489666921711585879E-001 * pmin**2 - 0.71096939603728015466544 * ramp_rate**2 - 0.29267439739652886809829E-001 * min_up_time**2 - 0.21596434569603947806193 * marg_cst**2 + 0.45778178995229547310064E-001 * no_load_cst**2 + 0.45925510249420986585989 * st_time_hot**2 + 0.56335678661677102141425 * pmax*pmin + 1.2783331619372102494481 * pmax*ramp_rate + 0.20857730072726364534130E-001 * pmax*min_up_time - 0.78172612720284251341951E-001 * pmax*marg_cst - 0.41038368814124794037301E-001 * pmax*no_load_cst - 0.14375665684415286915687 * pmax*st_time_hot + 0.95506821336977257286271E-001 * pmax*st_time_warm - 0.11482887396768459015162E-001 * pmax*st_time_cold - 0.74946437812786481647009 * pmin*ramp_rate - 0.20726200565198030917324E-001 * pmin*min_up_time + 0.33051412901530442811637E-001 * pmin*no_load_cst + 0.17664523456174380938721E-001 * pmin*st_time_hot + 0.73333965407797352398234E-001 * pmin*st_time_warm + 0.55411325543614541391801E-002 * pmin*st_time_cold - 0.14993782270991296012852 * pmin*st_cst_warm + 0.14201096653655609358680 * ramp_rate*st_time_hot - 0.79266229302067475748217E-001 * ramp_rate*st_time_warm + 0.25846934396561570268513E-001 * ramp_rate*st_time_cold - 0.61061926686306798595094E-001 * ramp_rate*st_cst_hot - 0.29086552849491433686557E-001 * min_up_time*marg_cst - 0.25392041430548149827384E-001 * min_up_time*no_load_cst + 0.25131525894921503266888E-001 * min_up_time*st_time_hot - 0.61296602937177473158226E-001 * min_up_time*st_time_warm - 0.24650130219600250441880E-001 * min_up_time*st_time_cold + 0.13613055631046833138242 * min_up_time*st_cst_hot - 0.81630396264451565713216E-001 * min_down_time*st_time_hot + 0.57715554688120770943094E-001 * min_down_time*st_time_warm - 0.44270896575604096190304E-001 * marg_cst*no_load_cst + 0.14602694782594320341285E-001 * marg_cst*st_time_hot + 0.23620934071643586577016E-001 * marg_cst*st_time_warm + 0.30100513374983199621493E-001 * marg_cst*st_time_cold - 0.21142065833489243065202 * marg_cst*st_cst_hot + 0.37461085885605371892382E-001 * no_load_cst*st_time_hot + 0.87824498108380454758359E-002 * no_load_cst*st_time_warm + 0.28173129830235205117717E-001 * no_load_cst*st_time_cold - 0.11074247546658426422095 * no_load_cst*st_cst_hot + 0.13469016470418419817445E-001 * (pmax*ramp_rate)**2 + 0.77203374355081164082115E-001 * (pmax*st_time_hot)**2 + 0.12854411556022580023773E-001 * (pmax*st_time_cold)**2 - 0.74117804214916468974472E-001 * (pmin*st_time_hot)**2 - 0.13805949057360652337811E-001 * (pmin*st_time_cold)**2 - 0.49684904352939138205691E-001 * (ramp_rate*st_time_hot)**2 + 0.27225405249701760401759E-001 * (ramp_rate*st_time_warm)**2 + 0.13401129734699141274334E-001 * (ramp_rate*st_time_cold)**2 - 0.47854419059223454735452E-001 * (ramp_rate*st_cst_hot)**2 + 0.35600669612539506680982E-001 * (min_up_time*st_time_hot)**2 - 0.19223070945139954235348E-001 * (marg_cst*st_cst_hot)**2 - 0.44264593615724469322092E-001 * (no_load_cst*st_time_hot)**2 

    z_unscale = z*zstd4 + zm4
    return z_unscale

with open(this_file_dir()+"/hours_zone_5.pkl", "rb") as input_file:
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

    z = 0.71481271173865656987090 * pmax - 0.48111466473211883521088 * pmin - 1.1652322194805291299957 * ramp_rate - 0.10420621210815696200402E-001 * min_up_time + 0.22738782732522513918560E-001 * min_down_time - 0.18204082007688954236713 * marg_cst - 0.72740650308384280364749E-001 * no_load_cst + 0.53187379852336021113235E-001 * st_time_hot - 0.52496107935146455250597E-002 * st_time_warm - 0.51682918400241126266614E-001 * st_time_cold - 0.95660479354265345874886E-001 * st_cst_hot - 0.74171129068158009323142E-001 * st_cst_warm + 0.21429629218005358248789 * pmax**2 + 0.22902984441792637038127 * pmin**2 + 1.1506081103879666205358 * ramp_rate**2 - 0.94176407999330344344990E-002 * min_up_time**2 - 0.99219969581520774115635E-002 * min_down_time**2 - 0.32466021436816799861091 * marg_cst**2 + 0.41307392600604554006027E-002 * no_load_cst**2 - 0.18963290185608203120005 * st_time_hot**2 - 0.40995391127423280730468 * pmax*pmin - 1.1362537638106073867306 * pmax*ramp_rate + 0.88315447043170722252192E-001 * pmax*min_up_time - 0.23872499734077518018793 * pmax*marg_cst - 0.19534043367597653251266 * pmax*no_load_cst - 0.28104603219437929162439 * pmax*st_time_hot + 0.12283626621874571549498 * pmax*st_time_warm - 0.78007081951411205711366E-001 * pmax*st_time_cold + 0.12502961420465885655773 * pmax*st_cst_hot + 0.87449474466275889383837 * pmin*ramp_rate - 0.52223163376444767247442E-001 * pmin*min_up_time + 0.81479509247652695536068E-001 * pmin*marg_cst + 0.13602850020181442292078 * pmin*no_load_cst + 0.15034793033648408133729 * pmin*st_time_hot - 0.53972004534839058964568E-001 * pmin*st_time_warm + 0.45979420765013580274516E-001 * pmin*st_time_cold - 0.10490873587405966094188 * pmin*st_cst_hot - 0.98386917274212101758302E-001 * ramp_rate*min_up_time - 0.14980459102764095632998E-001 * ramp_rate*min_down_time + 0.28589060327221382795670 * ramp_rate*marg_cst + 0.22132145907377540061667 * ramp_rate*no_load_cst + 0.38153755025067337403044 * ramp_rate*st_time_hot - 0.16294891336520220259665 * ramp_rate*st_time_warm + 0.10228883855705613037390 * ramp_rate*st_time_cold - 0.16042364133614353538526 * ramp_rate*st_cst_hot - 0.88554841178553438119092E-002 * min_up_time*marg_cst - 0.86636030203175624919698E-002 * min_up_time*no_load_cst + 0.50663343705487839593871E-002 * min_up_time*st_time_warm + 0.10204344526421627384338E-001 * min_down_time*st_time_cold - 0.65835563988063255935757E-001 * marg_cst*no_load_cst + 0.24291124728956024758642E-001 * marg_cst*st_time_hot + 0.19839762656960594801314E-002 * marg_cst*st_time_warm + 0.20814744492919850377133E-001 * marg_cst*st_time_cold - 0.15666562394343708231048 * marg_cst*st_cst_hot - 0.39589580006651768628867E-001 * no_load_cst*st_time_hot + 0.85824511788340648865692E-001 * no_load_cst*st_time_warm - 0.94095088197799658935594E-001 * no_load_cst*st_cst_warm - 0.78060365728168101817630E-001 * (pmax*ramp_rate)**2 + 0.21286010284619131388562E-001 * (pmax*st_time_hot)**2 + 0.19730664312794004427998E-001 * (pmax*st_cst_hot)**2 - 0.21930960959659896031271E-001 * (pmin*ramp_rate)**2 - 0.25881703101621474083505E-001 * (pmin*st_time_hot)**2 + 0.11024626231665378764402E-001 * (ramp_rate*min_up_time)**2 - 0.14192437893160724654562E-001 * (ramp_rate*no_load_cst)**2 + 0.19395675283511843584039E-001 * (ramp_rate*st_time_hot)**2 + 0.29458964965183593448517E-001 * (ramp_rate*st_time_warm)**2 + 0.27747855477523161726872E-001 * (ramp_rate*st_time_cold)**2 - 0.69901690125035151335275E-001 * (ramp_rate*st_cst_hot)**2 + 0.20387674165819290317270E-001 * (marg_cst*no_load_cst)**2 + 0.49866836104674620855359E-001 * (marg_cst*st_time_hot)**2 - 0.16404847847147369793719E-001 * (marg_cst*st_time_warm)**2 - 0.73684240002375020126002E-002 * (marg_cst*st_time_cold)**2 + 0.48828770323097105132693E-001 * (marg_cst*st_cst_hot)**2 

    z_unscale = z*zstd5 + zm5
    return z_unscale

with open(this_file_dir()+"/hours_zone_6.pkl", "rb") as input_file:
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

    z = 0.71462486330898289388358 * pmax - 0.53572136115670310196890 * pmin - 0.22764957332193741512683 * ramp_rate + 0.37591971582924176009666E-001 * min_up_time + 0.48282431540666977121123E-001 * min_down_time - 0.52332501480250892544888 * marg_cst - 0.20924079331580394680579 * no_load_cst - 0.37959632433541901086471 * st_time_hot + 0.14805075585464966270166 * st_time_warm - 0.59031356329030677809744E-001 * st_time_cold + 0.14931690667766830959984 * st_cst_hot + 0.87453577601660906215564E-001 * pmax**2 + 0.11482594966808733205621 * pmin**2 + 0.10644967700423899059636 * ramp_rate**2 + 0.18964337611856577070490E-001 * min_up_time**2 + 0.71647956382402946604038E-003 * min_down_time**2 - 0.38842294743456534655834 * marg_cst**2 + 0.20407429422024350729181E-001 * no_load_cst**2 + 0.19406893325723700849395 * st_time_hot**2 - 0.48429123511487065512071E-001 * pmax*pmin - 0.20776261948392732437085 * pmax*ramp_rate - 0.96810222426531155476281E-002 * pmax*min_up_time - 0.18848363484431554515375 * pmax*marg_cst - 0.10314970159515125802674 * pmax*no_load_cst + 0.26811750824755880423522E-001 * pmax*st_time_hot - 0.13547464989524837450374E-001 * pmax*st_time_warm + 0.11755704056577021346386E-001 * pmax*st_time_cold - 0.82279061583064597162895E-001 * pmax*st_cst_hot + 0.11541144587390782250136 * pmin*ramp_rate + 0.50532784482072799076002E-002 * pmin*min_up_time + 0.27354985331171557255692E-002 * pmin*marg_cst + 0.10994312950724370592237 * pmin*no_load_cst - 0.24113844422781194332117E-001 * pmin*st_time_hot + 0.17123586215265326065937E-001 * pmin*st_time_warm + 0.24672918968676727652367E-001 * pmin*st_cst_warm + 0.28071579994651651990267E-001 * ramp_rate*min_up_time + 0.41349691436494327712037E-001 * ramp_rate*marg_cst + 0.81675196442368754728980E-002 * ramp_rate*no_load_cst - 0.38300654850097889600469E-002 * ramp_rate*st_time_hot - 0.86500896086644088467388E-001 * ramp_rate*st_time_warm + 0.16029557599444838045244 * ramp_rate*st_cst_warm + 0.24439746542895404018969E-001 * min_up_time*marg_cst + 0.15079976139766524886876E-001 * min_up_time*no_load_cst + 0.73441913151007981452878E-001 * min_up_time*st_time_hot - 0.51119172235974064122388E-001 * min_up_time*st_time_warm + 0.18223485234639895391506E-001 * min_up_time*st_time_cold - 0.31997667808953786372506E-001 * min_up_time*st_cst_hot - 0.11064971831524682277514E-001 * min_down_time*marg_cst + 0.38632857590005079129192E-001 * min_down_time*st_time_hot - 0.14481751797920400237629 * marg_cst*no_load_cst + 0.53424758798663943282214E-001 * marg_cst*st_time_hot - 0.40767474331657164998699E-001 * marg_cst*st_time_warm + 0.73203943094293927318361E-002 * marg_cst*st_time_cold - 0.10308557630578546460143 * marg_cst*st_cst_hot - 0.48798725381759715247210E-001 * no_load_cst*st_time_hot + 0.37270459434693196942856E-001 * no_load_cst*st_time_warm - 0.42279721045557390468606E-001 * (pmax*pmin)**2 + 0.90226693719616512384674E-002 * (pmax*ramp_rate)**2 - 0.95790837204117358638644E-002 * (pmax*no_load_cst)**2 + 0.81398651784830959587946E-002 * (pmin*ramp_rate)**2 + 0.15851063393820044589155E-001 * (pmin*marg_cst)**2 - 0.44072690909438211673277E-002 * (ramp_rate*min_up_time)**2 - 0.14815771231923355669147E-001 * (ramp_rate*marg_cst)**2 + 0.15644707485723902823382E-001 * (ramp_rate*no_load_cst)**2 + 0.12564325975073422497763E-001 * (ramp_rate*st_time_hot)**2 + 0.84878827751342193763673E-002 * (ramp_rate*st_cst_hot)**2 - 0.78450464802398266889183E-002 * (min_up_time*marg_cst)**2 - 0.11936791735598088065595E-001 * (min_up_time*st_time_hot)**2 - 0.50857370220577227323822E-002 * (min_down_time*marg_cst)**2 - 0.31850942354293572920942E-002 * (min_down_time*st_cst_hot)**2 - 0.64332636848364486634178E-002 * (min_down_time*st_cst_warm)**2 + 0.63822804784012421297490E-001 * (marg_cst*no_load_cst)**2 + 0.29283206999893656580225E-001 * (marg_cst*st_time_hot)**2 - 0.69822023946792086124269E-001 * (marg_cst*st_time_warm)**2 - 0.53480230852528252982747E-001 * (marg_cst*st_time_cold)**2 + 0.14425145733158281635689 * (marg_cst*st_cst_hot)**2 - 0.57816812278829635141086E-001 * (no_load_cst*st_time_hot)**2 - 0.23788829083680934628475E-001 * (no_load_cst*st_time_warm)**2 - 0.26023298235199394945472E-001 * (no_load_cst*st_time_cold)**2 + 0.54494050287138495747286E-001 * (no_load_cst*st_cst_hot)**2 

    z_unscale = z*zstd6 + zm6
    return z_unscale

with open(this_file_dir()+"/hours_zone_7.pkl", "rb") as input_file:
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

    z = 0.73767992901703127728297 * pmax - 0.52021657942275334463034 * pmin - 0.22700501919653326421411 * ramp_rate + 0.38360799344691470791346E-001 * min_up_time + 0.36604588785496980363199E-001 * min_down_time - 0.52511085692885461906343 * marg_cst - 0.20414161003697786633815 * no_load_cst - 0.34750171694645870523743 * st_time_hot + 0.10788333891389541296935 * st_time_warm - 0.42792092129276211587285E-001 * st_time_cold + 0.18344901753546380418491 * st_cst_hot + 0.44766577393186617050969E-001 * pmax**2 + 0.42478557799294838770443E-001 * pmin**2 + 0.88093632795451806694054E-001 * ramp_rate**2 + 0.11061916502440179502220E-001 * min_up_time**2 - 0.70854962232066243457695E-002 * min_down_time**2 - 0.37089765701183668999974 * marg_cst**2 - 0.61982792279025887560717E-001 * no_load_cst**2 + 0.26331430391952781455700 * st_time_hot**2 - 0.23328240692916873272322E-001 * pmax*pmin - 0.11894180162858508653656 * pmax*ramp_rate - 0.19586016802935643510430 * pmax*marg_cst - 0.90296005754089886385927E-001 * pmax*no_load_cst + 0.12076661443968558784512 * pmax*st_time_hot - 0.11141986739535511552468 * pmax*st_time_warm - 0.85487297227554697592877E-002 * pmax*st_time_cold + 0.75957159921279521208604E-001 * pmin*ramp_rate + 0.24638138983013303101588E-001 * pmin*marg_cst + 0.93269965239525659539410E-001 * pmin*no_load_cst - 0.97967407456182611924511E-001 * pmin*st_time_hot + 0.11157749717129457345255 * pmin*st_time_warm - 0.48794704351046130907932E-001 * pmin*st_cst_warm + 0.14545726395527404373187E-001 * ramp_rate*min_up_time + 0.26529080071698536019698E-001 * ramp_rate*marg_cst - 0.14612347951201698492696 * ramp_rate*st_time_hot + 0.11322096929847415280879 * ramp_rate*st_time_warm + 0.15552342491199034821281E-001 * min_up_time*marg_cst + 0.16000849936879703305781E-001 * min_up_time*no_load_cst + 0.92074409320438671544551E-001 * min_up_time*st_time_hot - 0.69209088014681091616254E-001 * min_up_time*st_time_warm + 0.95982761501211856514093E-002 * min_up_time*st_time_cold + 0.12414453545909025222738E-001 * min_down_time*st_time_hot - 0.14396995688918412525936 * marg_cst*no_load_cst + 0.43489780812303188706203E-001 * marg_cst*st_time_hot - 0.35211497742601349791869E-001 * marg_cst*st_time_warm + 0.67102188290565946415800E-002 * marg_cst*st_time_cold - 0.10906315784710120253553 * marg_cst*st_cst_hot - 0.41412822182506345281006E-001 * no_load_cst*st_time_hot + 0.33285065273615285774689E-001 * no_load_cst*st_time_warm - 0.46264712339248934003244E-002 * (pmax*marg_cst)**2 - 0.98510624114759044039902E-002 * (pmax*no_load_cst)**2 - 0.12035421309945053269796E-001 * (pmin*marg_cst)**2 - 0.12372260011944118812677E-001 * (ramp_rate*marg_cst)**2 + 0.14088397395182394519764E-001 * (ramp_rate*no_load_cst)**2 + 0.17144696011354966513895E-001 * (ramp_rate*st_time_hot)**2 - 0.75802279322207026870939E-002 * (min_up_time*marg_cst)**2 - 0.12377872890750376069779E-001 * (min_up_time*st_time_hot)**2 - 0.46558539509785144133724E-002 * (min_down_time*marg_cst)**2 + 0.67782086189136961373869E-001 * (marg_cst*no_load_cst)**2 + 0.19469989541937630234125E-001 * (marg_cst*st_time_hot)**2 - 0.71203958945225437338067E-001 * (marg_cst*st_time_warm)**2 - 0.56369092625652615635712E-001 * (marg_cst*st_time_cold)**2 + 0.14720740868626852604351 * (marg_cst*st_cst_hot)**2 - 0.17346216696974853938018E-001 * (no_load_cst*st_time_hot)**2 + 0.33410884592574296991785E-001 * (no_load_cst*st_cst_hot)**2 


    z_unscale = z*zstd7 + zm7
    return z_unscale

with open(this_file_dir()+"/hours_zone_8.pkl", "rb") as input_file:
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

    z = 0.69931033153141020086707 * pmax - 0.47451823899652750826661 * pmin - 0.54257514793064265390399 * ramp_rate - 0.81406863282342950549619E-002 * min_up_time - 0.25400279870363140499734E-001 * min_down_time - 0.22265864583624978001453 * marg_cst - 0.76636486372387194188072E-001 * no_load_cst - 0.59220550145998557267646E-001 * st_time_hot - 0.34890927064606488960941E-001 * st_time_warm + 0.66189946655319409352103E-001 * st_time_cold + 0.66148478369876639870739E-001 * st_cst_hot + 0.63513737759543109628879E-001 * st_cst_cold - 0.51870784994946439461216 * pmax**2 - 0.18618908246469695333403 * pmin**2 - 0.70963626291243642185691 * ramp_rate**2 + 0.87069914605585815858824E-002 * min_up_time**2 - 0.65686292231521753448575E-001 * min_down_time**2 - 0.37797947672948950526006 * marg_cst**2 + 0.30261160285540724367015E-003 * no_load_cst**2 + 0.42412430581326993062063 * st_time_hot**2 + 0.70074326504225059597530 * pmax*pmin + 1.2976627598313585920664 * pmax*ramp_rate - 0.56898441431990035010990E-001 * pmax*min_up_time - 0.55778432420070120367761E-001 * pmax*min_down_time + 0.59136288069806135225814E-001 * pmax*marg_cst + 0.87284451643933169684431E-001 * pmax*no_load_cst + 0.24025157314263442809299 * pmax*st_time_hot - 0.71495175639411379098220E-001 * pmax*st_time_warm + 0.83406132394285806275214E-001 * pmax*st_time_cold - 0.38695168198395257785904 * pmax*st_cst_cold - 0.81267965539180164746824 * pmin*ramp_rate + 0.30894048283917618924477E-001 * pmin*min_up_time + 0.23436476844449490725131E-001 * pmin*min_down_time - 0.92110998679183278547988E-001 * pmin*marg_cst - 0.28282076889233568467441E-001 * pmin*no_load_cst - 0.92525144577430579184707E-001 * pmin*st_time_hot - 0.51107119605671456896712E-001 * pmin*st_time_warm + 0.19749202147296751608252 * pmin*st_cst_warm + 0.75630953067338996254065E-001 * ramp_rate*min_up_time + 0.82398570280680372146520E-001 * ramp_rate*min_down_time - 0.11768082636298823895338 * ramp_rate*marg_cst - 0.13323402704936163076788 * ramp_rate*no_load_cst - 0.51696031057526214969045 * ramp_rate*st_time_hot + 0.21505495067561489208607 * ramp_rate*st_time_warm - 0.14738389860613052007032 * ramp_rate*st_time_cold + 0.45527637268316167862281 * ramp_rate*st_cst_hot - 0.10443786998484078445970E-001 * min_up_time*no_load_cst + 0.56994352046973563788446E-002 * min_up_time*st_time_hot - 0.63751644680448684918139E-002 * min_up_time*st_time_warm + 0.97920598831279220125579E-002 * min_up_time*st_cst_hot + 0.12871854750684491586199 * min_down_time*st_time_hot - 0.25740446442960457362759 * min_down_time*st_time_warm + 0.28584515349280020801714 * min_down_time*st_cst_warm - 0.85669472712426877891723E-001 * marg_cst*no_load_cst + 0.32155008972817256118759E-001 * marg_cst*st_time_hot - 0.59871931706555869129316E-002 * marg_cst*st_time_warm + 0.20447307173445408712276E-001 * marg_cst*st_time_cold - 0.18449836757963691824536 * marg_cst*st_cst_hot - 0.37381063537477422775712E-001 * no_load_cst*st_time_hot + 0.10843264358892211463026 * no_load_cst*st_time_warm - 0.12966332494891946724991 * no_load_cst*st_cst_warm + 0.56148701489964732494475E-002 * (pmax*min_down_time)**2 - 0.21711807036759379896385E-001 * (pmax*marg_cst)**2 - 0.33520181955508189519666E-001 * (pmax*st_time_hot)**2 + 0.25897084455949646863981E-001 * (pmax*st_time_warm)**2 + 0.15609280383583492776700E-001 * (pmax*st_time_cold)**2 - 0.51967748026013199647011E-001 * (pmax*st_cst_hot)**2 - 0.23470261271643328565562E-001 * (pmin*ramp_rate)**2 + 0.50668625858246441051591E-002 * (pmin*st_time_hot)**2 - 0.12673932073645335222301E-001 * (ramp_rate*min_up_time)**2 - 0.78947189267525620659516E-002 * (ramp_rate*min_down_time)**2 + 0.14295589684665578611433E-001 * (ramp_rate*marg_cst)**2 + 0.15843199312911438597640E-001 * (ramp_rate*no_load_cst)**2 + 0.71204779770757439805173E-001 * (ramp_rate*st_time_hot)**2 - 0.58019431311728930644733E-001 * (ramp_rate*st_time_warm)**2 - 0.35174470737492150029802E-001 * (ramp_rate*st_time_cold)**2 + 0.99607295195970821377607E-001 * (ramp_rate*st_cst_hot)**2 - 0.35975563042885969685259E-004 * (min_up_time*st_time_hot)**2 + 0.44669029201679519291979E-001 * (min_down_time*st_time_hot)**2 + 0.93988151731117294124251E-002 * (min_down_time*st_time_warm)**2 + 0.14854041813840862876206E-001 * (min_down_time*st_time_cold)**2 + 0.17460767301570308956959E-001 * (marg_cst*no_load_cst)**2 + 0.61476415989439849030251E-001 * (marg_cst*st_time_hot)**2 + 0.71698817556130403638726E-002 * (marg_cst*st_time_cold)**2 + 0.20535098714982130685414E-001 * (marg_cst*st_cst_hot)**2 - 0.23838808898897580895726E-001 * (no_load_cst*st_time_hot)**2 

    z_unscale = z*zstd8 + zm8
    return z_unscale

with open(this_file_dir()+"/hours_zone_9.pkl", "rb") as input_file:
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

    z = 0.69489093278685043486576 * pmax - 0.52315742981481216933304 * pmin - 0.14083645817291454149789 * ramp_rate + 0.33851136062223043110553E-001 * min_up_time + 0.35253727805887508095495E-001 * min_down_time - 0.54515839898432538745965 * marg_cst - 0.20477480518220461669365 * no_load_cst - 0.41644082111213348573742 * st_time_hot + 0.15947087511387084135528 * st_time_warm - 0.28750399430292852831492E-001 * st_time_cold + 0.19643270614657487582733 * st_cst_hot - 0.19430712513630798987663 * pmax**2 + 0.26214337115957261903354E-001 * pmin**2 - 0.10331728720634322793437 * ramp_rate**2 + 0.46209776284385252476117E-002 * min_up_time**2 + 0.17714518498827640929916E-001 * min_down_time**2 - 0.38714740776662220911319 * marg_cst**2 - 0.37046066187689234705527E-001 * no_load_cst**2 + 0.38721279008132736265679 * st_time_hot**2 + 0.10434895539529528973866 * pmax*pmin + 0.22462794900170426903863 * pmax*ramp_rate - 0.21641503250702912031223 * pmax*marg_cst - 0.84952255216421510009006E-001 * pmax*no_load_cst - 0.60841629909778115192776E-001 * pmax*st_time_hot + 0.32124696431623155079560E-001 * pmax*st_time_warm - 0.74892205114465582732608E-002 * pmax*st_time_cold - 0.31144268942663918403602E-001 * pmax*st_cst_hot - 0.10411356095150524347925 * pmin*ramp_rate + 0.38191145758792059694553E-001 * pmin*marg_cst + 0.87095925155747877521861E-001 * pmin*no_load_cst + 0.17134069436802526475994E-001 * pmin*st_time_hot - 0.10367876052336617506699E-002 * pmin*st_time_warm + 0.75958575499480165960087E-001 * ramp_rate*marg_cst + 0.58216740653558918250354E-001 * ramp_rate*st_time_hot - 0.38289089318791734439795E-001 * ramp_rate*st_time_warm + 0.24694735574040376641092E-001 * min_up_time*marg_cst + 0.16002199477357838958147E-001 * min_up_time*no_load_cst + 0.75226808710527975909699E-001 * min_up_time*st_time_hot - 0.58107170331001135876736E-001 * min_up_time*st_time_warm + 0.11049179604766201645560E-001 * min_up_time*st_time_cold - 0.97485635612475762118434E-002 * min_down_time*marg_cst + 0.15274507832059717668538E-001 * min_down_time*st_time_hot - 0.14836151499915734008894 * marg_cst*no_load_cst + 0.39508640481557423529413E-001 * marg_cst*st_time_hot - 0.33319673410416486558638E-001 * marg_cst*st_time_warm + 0.60281013680550067299069E-002 * marg_cst*st_time_cold - 0.11014406540179352833597 * marg_cst*st_cst_hot - 0.38258586550003206849802E-001 * no_load_cst*st_time_hot + 0.30154776107972586923767E-001 * no_load_cst*st_time_warm + 0.10559241087720577789710E-001 * (pmax*ramp_rate)**2 + 0.17435567623891493060917E-001 * (pmax*marg_cst)**2 - 0.15438801130521023069475E-001 * (pmin*st_time_hot)**2 + 0.70036432040566217946398E-002 * (pmin*st_cst_cold)**2 - 0.17798403396014760013966E-001 * (ramp_rate*marg_cst)**2 + 0.13522060387860343477762E-001 * (ramp_rate*no_load_cst)**2 - 0.10498981357942516884352E-001 * (ramp_rate*st_time_hot)**2 - 0.73014013837265521675302E-002 * (min_up_time*min_down_time)**2 - 0.81828248756639944405666E-002 * (min_up_time*marg_cst)**2 - 0.48204463330118038286165E-002 * (min_down_time*marg_cst)**2 + 0.68523270622669704699170E-001 * (marg_cst*no_load_cst)**2 + 0.30877965356079145609058E-001 * (marg_cst*st_time_hot)**2 - 0.70459337892621085042144E-001 * (marg_cst*st_time_warm)**2 - 0.53362751711675128485357E-001 * (marg_cst*st_time_cold)**2 + 0.14571801892599145489271 * (marg_cst*st_cst_hot)**2 - 0.40109565701173847951377E-001 * (no_load_cst*st_time_hot)**2 - 0.35487187280471864381237E-001 * (no_load_cst*st_cst_hot)**2 + 0.57444777719000833515750E-001 * (no_load_cst*st_cst_cold)**2 

    z_unscale = z*zstd9 + zm9
    return z_unscale

with open(this_file_dir()+"/hours_zone_10.pkl", "rb") as input_file:
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

    z = - 0.88007140780954187797569E-001 * pmax - 0.81308877123764753541701E-002 * pmin + 0.72142424341750263638851E-001 * ramp_rate + 0.40065844721205038161949E-002 * min_up_time + 0.81387309920868070728384E-002 * min_down_time - 0.91217941313312034257166 * marg_cst - 0.58948733226901806059317E-001 * no_load_cst - 0.15372200629804877447526 * st_time_hot - 0.57031964995380244470846E-002 * st_time_warm + 0.69694063695635119493055E-001 * st_time_cold + 0.92966795635905902228657E-001 * st_cst_hot + 0.10484208453737227373370 * st_cst_warm + 0.22754695944576228128220E-001 * pmax**2 + 0.59432115477443432649540E-002 * pmin**2 - 0.81717432218728698785359E-003 * ramp_rate**2 - 0.15740054993947400364257E-002 * min_up_time**2 - 0.35223911896825059637817E-002 * min_down_time**2 - 0.42397208170980343888701 * marg_cst**2 - 0.26769252008289598498969E-001 * no_load_cst**2 + 0.40793462578187411615716 * st_time_hot**2 - 0.20276483678236367769188E-001 * pmax*pmin - 0.36820261460468066416762E-001 * pmax*ramp_rate + 0.16134296324316937248922E-001 * pmax*marg_cst - 0.46305129961763972709643E-002 * pmax*st_time_hot + 0.72022884471070877307475E-002 * pmax*st_time_warm + 0.16366167446852642602950E-001 * pmin*ramp_rate - 0.18062512842871854962246E-001 * pmin*marg_cst + 0.70680691267442155578671E-002 * pmin*no_load_cst + 0.44588070431590080128226E-002 * pmin*st_time_hot - 0.83683437163822906035282E-002 * pmin*st_time_warm - 0.32835585164817460493425E-002 * ramp_rate*no_load_cst + 0.20561905656413814041938E-001 * ramp_rate*st_time_hot - 0.18435092353989157576422E-001 * ramp_rate*st_time_warm + 0.61888168340458044047736E-002 * min_up_time*st_time_hot - 0.45652588975197729015409E-001 * marg_cst*no_load_cst + 0.70000876845662000566572E-002 * marg_cst*st_time_hot - 0.77562781030912119131782E-002 * marg_cst*st_time_warm + 0.11225299720508512008738E-002 * marg_cst*st_time_cold - 0.50554044974295976599965E-001 * marg_cst*st_cst_hot - 0.32906771165950462164351E-002 * no_load_cst*st_time_warm + 0.36731765000431715849361E-002 * (pmax*ramp_rate)**2 + 0.72354013322980777547788E-002 * (pmin*marg_cst)**2 + 0.22838878830913557534121E-002 * (ramp_rate*marg_cst)**2 - 0.54445386008780370656890E-002 * (ramp_rate*st_time_hot)**2 + 0.23006266698437725232163E-001 * (marg_cst*no_load_cst)**2 + 0.99550526062851998204684E-002 * (marg_cst*st_time_hot)**2 - 0.14750175893296814041977E-001 * (marg_cst*st_time_warm)**2 - 0.11752031576057551942593E-001 * (marg_cst*st_time_cold)**2 + 0.39403083936096656991754E-001 * (marg_cst*st_cst_hot)**2 


    z_unscale = z*zstd10 + zm10
    return z_unscale
