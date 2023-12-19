atts_folk = [
    "AGEP",
    "SCHL",
    "MAR",
    "SEX",
    "DIS",
    "ESP",
    "CIT",
    "MIG",
    "MIL",
    "ANC",
    "NATIVITY",
    "DEAR",
    "DEYE",
    "DREM",
    "PINCP",
    "ESR",
    "ST",
    "FER",
    "RAC1P",
]

dict_atts = {
    "adult": [
        "age",
        "workclass",
        "fnlwgt",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital_gain",
        "capital_loss",
        "work_hours",
        "native_country",
    ],
    "aloi": ["att_{}".format(el) for el in range(1, 129)],
    "giveme": [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ],
    "shuttle": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
    "phishing": [
        "having_IP_Address",
        "URL_Length",
        "Shortining_Service",
        "having_At_Symbol",
        "double_slash_redirecting",
        "Prefix_Suffix",
        "having_Sub_Domain",
        "SSLfinal_State",
        "Domain_registeration_length",
        "Favicon",
        "port",
        "HTTPS_token",
        "Request_URL",
        "URL_of_Anchor",
        "Links_in_tags",
        "SFH",
        "Submitting_to_email",
        "Abnormal_URL",
        "Redirect",
        "on_mouseover",
        "RightClick",
        "popUpWidnow",
        "Iframe",
        "age_of_domain",
        "DNSRecord",
        "web_traffic",
        "Page_Rank",
        "Google_Index",
        "Links_pointing_to_page",
        "Statistical_report",
    ],
    "bank": [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    ],
    "mammography": ["attr1", "attr2", "attr3", "attr4", "attr5", "attr6"],
    "helena": ["V{}".format(el) for el in range(1, 28)],
    "higgs": [
        "lepton_pT",
        "lepton_eta",
        "lepton_phi",
        "missing_energy_magnitude",
        "missing_energy_phi",
        "jet1pt",
        "jet1eta",
        "jet1phi",
        "jet1b-tag",
        "jet2pt",
        "jet2eta",
        "jet2phi",
        "jet2b-tag",
        "jet3pt",
        "jet3eta",
        "jet3phi",
        "jet3b-tag",
        "jet4pt",
        "jet4eta",
        "jet4phi",
        "jet4b-tag",
        "m_jj",
        "m_jjj",
        "m_lv",
        "m_jlv",
        "m_bb",
        "m_wbb",
        "m_wwbb",
    ],
    "indian": ["X_{}".format(str(col).zfill(3)) for col in range(0, 220)],
    "jannis": ["V{}".format(str(el)) for el in range(1, 55)],
    "epsilon": ["X_{}".format(str(i).zfill(4)) for i in range(1, 2001)],
    "miniboone": ["ParticleID_{}".format(str(el)) for el in range(0, 50)],
    "covtype": [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        "Wilderness_Area1",
        "Wilderness_Area2",
        "Wilderness_Area3",
        "Wilderness_Area4",
        "Soil_Type1",
        "Soil_Type2",
        "Soil_Type3",
        "Soil_Type4",
        "Soil_Type5",
        "Soil_Type6",
        "Soil_Type7",
        "Soil_Type8",
        "Soil_Type9",
        "Soil_Type10",
        "Soil_Type11",
        "Soil_Type12",
        "Soil_Type13",
        "Soil_Type14",
        "Soil_Type15",
        "Soil_Type16",
        "Soil_Type17",
        "Soil_Type18",
        "Soil_Type19",
        "Soil_Type20",
        "Soil_Type21",
        "Soil_Type22",
        "Soil_Type23",
        "Soil_Type24",
        "Soil_Type25",
        "Soil_Type26",
        "Soil_Type27",
        "Soil_Type28",
        "Soil_Type29",
        "Soil_Type30",
        "Soil_Type31",
        "Soil_Type32",
        "Soil_Type33",
        "Soil_Type34",
        "Soil_Type35",
        "Soil_Type36",
        "Soil_Type37",
        "Soil_Type38",
        "Soil_Type39",
        "Soil_Type40",
    ],
    "ucicredit": [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
        "x17",
        "x18",
        "x19",
        "x20",
        "x21",
        "x22",
        "x23",
    ],
    "compass": [
        "sex",
        "age",
        "age_cat",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "days_b_screening_arrest",
        "c_days_from_compas",
        "c_charge_degree",
        "decile_score.1",
        "score_text",
        "v_type_of_assessment",
        "v_decile_score",
        "v_score_text",
        "end",
    ],
    "newspaper": [
        "HH Income",
        "Home Ownership",
        "Ethnicity",
        "dummy for Children",
        "Year Of Residence",
        "Age range",
        "Language",
        "Address",
        "State",
        "City",
        "County",
        "Zip Code",
        "weekly fee",
        "Deliveryperiod",
        "Nielsen Prizm",
        "reward program",
        "Source Channel",
    ],
    "house": [
        "P1",
        "P5p1",
        "P6p2",
        "P11p4",
        "P14p9",
        "P15p1",
        "P15p3",
        "P16p2",
        "P18p2",
        "P27p4",
        "H2p2",
        "H8p2",
        "H10p1",
        "H13p1",
        "H18pA",
        "H40p4",
    ],
    "online": [
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "Browser",
        "OperatingSystems",
        "VisitorType",
        "Administrative_Duration",
        "Month",
        "Region",
        "ProductRelated",
        "ProductRelated_Duration",
        "Informational",
        "TrafficType",
        "Informational_Duration",
        "Weekend",
        "Administrative",
    ],
    "heloc": [
        "ExternalRiskEstimate",
        "MSinceOldestTradeOpen",
        "MSinceMostRecentTradeOpen",
        "AverageMInFile",
        "NumSatisfactoryTrades",
        "NumTrades60Ever2DerogPubRec",
        "NumTrades90Ever2DerogPubRec",
        "PercentTradesNeverDelq",
        "MSinceMostRecentDelq",
        "MaxDelq2PublicRecLast12M",
        "NumTotalTrades",
        "NumTradesOpeninLast12M",
        "PercentInstallTrades",
        "MSinceMostRecentInqexcl7days",
        "NumInqLast6M",
        "NumInqLast6Mexcl7days",
        "NetFractionRevolvingBurden",
        "NetFractionInstallBurden",
        "NumRevolvingTradesWBalance",
        "NumInstallTradesWBalance",
        "NumBank2NatlTradesWHighUtilization",
        "PercentTradesWBalance",
    ],
    "letter": [
        "x-box",
        "y-box",
        "width",
        "high",
        "onpix",
        "x-bar",
        "y-bar",
        "x2bar",
        "y2bar",
        "xybar",
        "x2ybr",
        "xy2br",
        "x-ege",
        "xegvy",
        "y-ege",
        "yegvx",
    ],
    "electricity": [
        "date",
        "period",
        "nswprice",
        "nswdemand",
        "vicprice",
        "vicdemand",
        "transfer",
    ],
    "phoneme": ["V1", "V2", "V3", "V4", "V5"],
    "kddipums97": [
        "value",
        "rent",
        "ftotinc",
        "momloc",
        "famsize",
        "nchild",
        "eldch",
        "yngch",
        "nsibs",
        "age",
        "occscore",
        "sei",
        "inctot",
        "incwage",
        "incbus",
        "incfarm",
        "incss",
        "incwelfr",
        "incother",
        "poverty",
    ],
    "magic": [
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
    ],
    "rl": [
        "V1",
        "V5",
        "V6",
        "V8",
        "V14",
        "V15",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
    ],
    "upselling": [
        "Var6",
        "Var13",
        "Var21",
        "Var22",
        "Var24",
        "Var25",
        "Var28",
        "Var35",
        "Var38",
        "Var57",
        "Var65",
        "Var73",
        "Var74",
        "Var76",
        "Var78",
        "Var81",
        "Var83",
        "Var85",
        "Var109",
        "Var112",
        "Var113",
        "Var119",
        "Var123",
        "Var125",
        "Var126",
        "Var132",
        "Var133",
        "Var134",
        "Var140",
        "Var144",
        "Var149",
        "Var153",
        "Var160",
        "Var163",
        "Var196",
        "Var203",
        "Var205",
        "Var207",
        "Var208",
        "Var210",
        "Var211",
        "Var218",
        "Var221",
        "Var223",
        "Var227",
    ],
    "pol": [
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f13",
        "f14",
        "f15",
        "f16",
        "f17",
        "f18",
        "f19",
        "f20",
        "f21",
        "f22",
        "f23",
        "f24",
        "f25",
        "f26",
        "f27",
        "f28",
        "f29",
        "f30",
        "f31",
        "f32",
        "f33",
    ],
    "eye": [
        "lineNo",
        "assgNo",
        "P1stFixation",
        "P2stFixation",
        "prevFixDur",
        "firstfixDur",
        "firstPassFixDur",
        "nextFixDur",
        "firstSaccLen",
        "lastSaccLen",
        "prevFixPos",
        "landingPos",
        "leavingPos",
        "totalFixDur",
        "meanFixDur",
        "regressLen",
        "nextWordRegress",
        "regressDur",
        "pupilDiamMax",
        "pupilDiamLag",
        "timePrtctg",
        "titleNo",
        "wordNo",
    ],
    "ACSPublicCoverage2018AL": atts_folk,
    "ACSPublicCoverage2018AK": atts_folk,
    "ACSPublicCoverage2018AZ": atts_folk,
    "ACSPublicCoverage2018AR": atts_folk,
    "ACSPublicCoverage2018CA": atts_folk,
    "ACSPublicCoverage2018CO": atts_folk,
    "ACSPublicCoverage2018CT": atts_folk,
    "ACSPublicCoverage2018DE": atts_folk,
    "ACSPublicCoverage2018FL": atts_folk,
    "ACSPublicCoverage2018GA": atts_folk,
    "ACSPublicCoverage2018HI": atts_folk,
    "ACSPublicCoverage2018ID": atts_folk,
    "ACSPublicCoverage2018IL": atts_folk,
    "ACSPublicCoverage2018IN": atts_folk,
    "ACSPublicCoverage2018IA": atts_folk,
    "ACSPublicCoverage2018KS": atts_folk,
    "ACSPublicCoverage2018KY": atts_folk,
    "ACSPublicCoverage2018LA": atts_folk,
    "ACSPublicCoverage2018ME": atts_folk,
    "ACSPublicCoverage2018MD": atts_folk,
    "ACSPublicCoverage2018MA": atts_folk,
    "ACSPublicCoverage2018MI": atts_folk,
    "ACSPublicCoverage2018MN": atts_folk,
    "ACSPublicCoverage2018MS": atts_folk,
    "ACSPublicCoverage2018MO": atts_folk,
    "ACSPublicCoverage2018MT": atts_folk,
    "ACSPublicCoverage2018NE": atts_folk,
    "ACSPublicCoverage2018NV": atts_folk,
    "ACSPublicCoverage2018NH": atts_folk,
    "ACSPublicCoverage2018NJ": atts_folk,
    "ACSPublicCoverage2018NM": atts_folk,
    "ACSPublicCoverage2018NY": atts_folk,
    "ACSPublicCoverage2018NC": atts_folk,
    "ACSPublicCoverage2018ND": atts_folk,
    "ACSPublicCoverage2018OH": atts_folk,
    "ACSPublicCoverage2018OK": atts_folk,
    "ACSPublicCoverage2018OR": atts_folk,
    "ACSPublicCoverage2018PA": atts_folk,
    "ACSPublicCoverage2018RI": atts_folk,
    "ACSPublicCoverage2018SC": atts_folk,
    "ACSPublicCoverage2018SD": atts_folk,
    "ACSPublicCoverage2018TN": atts_folk,
    "ACSPublicCoverage2018TX": atts_folk,
    "ACSPublicCoverage2018UT": atts_folk,
    "ACSPublicCoverage2018VT": atts_folk,
    "ACSPublicCoverage2018VA": atts_folk,
    "ACSPublicCoverage2018WA": atts_folk,
    "ACSPublicCoverage2018WV": atts_folk,
    "ACSPublicCoverage2018WI": atts_folk,
    "ACSPublicCoverage2018WY": atts_folk,
    "ACSPublicCoverage2018PR": atts_folk,
}


dict_binary = {
    "adult": True,
    "aloi": False,
    "jannis": True,
    "helena": False,
    "catsdogs": True,
    "cifar10": False,
    "compass": True,
    "heloc": True,
    "letter": False,
    "house": True,
    "newspaper": True,
    "online": True,
    "ucicredit": True,
    "SVHN": False,
    "waterbirds": True,
    "stanfordcars": False,
    "MNIST": False,
    "food101": False,
    "indian": False,
    "bank": True,
    "mammography": True,
    "giveme": True,
    "phishing": True,
    "higgs": True,
    "shuttle": False,
    "covtype": False,
    "oxfordpets": True,
    "miniboone": True,
    "xray": True,
    "eye": True,
    "electricity": True,
    "pol": False,
    "upselling": True,
    "rl": True,
    "magic": True,
    "phoneme": True,
    "kddipums97": True,
    "organamnist": False,
    "organcmnist": False,
    "chestmnist": True,
    "breastmnist": True,
    "dermamnist": False,
    "bloodmnist": False,
    "pneumoniamnist": True,
    "tissuemnist": False,
    "octmnist": False,
    "pathmnist": False,
    "retinamnist": False,
    "organsmnist": False,
    "FashionMNIST": False,
    "cifar100": False,
    "ACSPublicCoverage2018AL": True,
    "ACSPublicCoverage2018AK": True,
    "ACSPublicCoverage2018AZ": True,
    "ACSPublicCoverage2018AR": True,
    "ACSPublicCoverage2018CA": True,
    "ACSPublicCoverage2018CO": True,
    "ACSPublicCoverage2018CT": True,
    "ACSPublicCoverage2018DE": True,
    "ACSPublicCoverage2018FL": True,
    "ACSPublicCoverage2018GA": True,
    "ACSPublicCoverage2018HI": True,
    "ACSPublicCoverage2018ID": True,
    "ACSPublicCoverage2018IL": True,
    "ACSPublicCoverage2018IN": True,
    "ACSPublicCoverage2018IA": True,
    "ACSPublicCoverage2018KS": True,
    "ACSPublicCoverage2018KY": True,
    "ACSPublicCoverage2018LA": True,
    "ACSPublicCoverage2018ME": True,
    "ACSPublicCoverage2018MD": True,
    "ACSPublicCoverage2018MA": True,
    "ACSPublicCoverage2018MI": True,
    "ACSPublicCoverage2018MN": True,
    "ACSPublicCoverage2018MS": True,
    "ACSPublicCoverage2018MO": True,
    "ACSPublicCoverage2018MT": True,
    "ACSPublicCoverage2018NE": True,
    "ACSPublicCoverage2018NV": True,
    "ACSPublicCoverage2018NH": True,
    "ACSPublicCoverage2018NJ": True,
    "ACSPublicCoverage2018NM": True,
    "ACSPublicCoverage2018NY": True,
    "ACSPublicCoverage2018NC": True,
    "ACSPublicCoverage2018ND": True,
    "ACSPublicCoverage2018OH": True,
    "ACSPublicCoverage2018OK": True,
    "ACSPublicCoverage2018OR": True,
    "ACSPublicCoverage2018PA": True,
    "ACSPublicCoverage2018RI": True,
    "ACSPublicCoverage2018SC": True,
    "ACSPublicCoverage2018SD": True,
    "ACSPublicCoverage2018TN": True,
    "ACSPublicCoverage2018TX": True,
    "ACSPublicCoverage2018UT": True,
    "ACSPublicCoverage2018VT": True,
    "ACSPublicCoverage2018VA": True,
    "ACSPublicCoverage2018WA": True,
    "ACSPublicCoverage2018WV": True,
    "ACSPublicCoverage2018WI": True,
    "ACSPublicCoverage2018WY": True,
    "ACSPublicCoverage2018PR": True,
}

dict_epochs = {
    "adult": 300,
    "aloi": 300,
    "bank": 300,
    "compass": 300,
    "heloc": 300,
    "letter": 300,
    "house": 300,
    "newspaper": 300,
    "online": 300,
    "ucicredit": 300,
    "xray": 300,
    "catsdogs": 300,
    "celeba": 50,
    "cifar10": 300,
    "covtype": 300,
    "food101": 300,
    "giveme": 300,
    "helena": 300,
    "higgs": 300,
    "indian": 300,
    "jannis": 300,
    "mammography": 300,
    "miniboone": 300,
    "MNIST": 300,
    "phishing": 300,
    "shuttle": 300,
    "stanfordcars": 300,
    "SVHN": 300,
    "waterbirds": 300,
    "oxfordpets": 300,
    "eye": 300,
    "electricity": 300,
    "pol": 300,
    "upselling": 300,
    "rl": 300,
    "magic": 300,
    "phoneme": 300,
    "kddipums97": 300,
    "organamnist": 300,
    "organcmnist": 300,
    "chestmnist": 300,
    "breastmnist": 300,
    "dermamnist": 300,
    "bloodmnist": 300,
    "pneumoniamnist": 300,
    "tissuemnist": 300,
    "octmnist": 300,
    "pathmnist": 300,
    "retinamnist": 300,
    "organsmnist": 300,
    "FashionMNIST": 300,
    "cifar100": 300,
    "ACSPublicCoverage2018AL": 300,
    "ACSPublicCoverage2018AK": 300,
    "ACSPublicCoverage2018AZ": 300,
    "ACSPublicCoverage2018AR": 300,
    "ACSPublicCoverage2018CA": 300,
    "ACSPublicCoverage2018CO": 300,
    "ACSPublicCoverage2018CT": 300,
    "ACSPublicCoverage2018DE": 300,
    "ACSPublicCoverage2018FL": 300,
    "ACSPublicCoverage2018GA": 300,
    "ACSPublicCoverage2018HI": 300,
    "ACSPublicCoverage2018ID": 300,
    "ACSPublicCoverage2018IL": 300,
    "ACSPublicCoverage2018IN": 300,
    "ACSPublicCoverage2018IA": 300,
    "ACSPublicCoverage2018KS": 300,
    "ACSPublicCoverage2018KY": 300,
    "ACSPublicCoverage2018LA": 300,
    "ACSPublicCoverage2018ME": 300,
    "ACSPublicCoverage2018MD": 300,
    "ACSPublicCoverage2018MA": 300,
    "ACSPublicCoverage2018MI": 300,
    "ACSPublicCoverage2018MN": 300,
    "ACSPublicCoverage2018MS": 300,
    "ACSPublicCoverage2018MO": 300,
    "ACSPublicCoverage2018MT": 300,
    "ACSPublicCoverage2018NE": 300,
    "ACSPublicCoverage2018NV": 300,
    "ACSPublicCoverage2018NH": 300,
    "ACSPublicCoverage2018NJ": 300,
    "ACSPublicCoverage2018NM": 300,
    "ACSPublicCoverage2018NY": 300,
    "ACSPublicCoverage2018NC": 300,
    "ACSPublicCoverage2018ND": 300,
    "ACSPublicCoverage2018OH": 300,
    "ACSPublicCoverage2018OK": 300,
    "ACSPublicCoverage2018OR": 300,
    "ACSPublicCoverage2018PA": 300,
    "ACSPublicCoverage2018RI": 300,
    "ACSPublicCoverage2018SC": 300,
    "ACSPublicCoverage2018SD": 300,
    "ACSPublicCoverage2018TN": 300,
    "ACSPublicCoverage2018TX": 300,
    "ACSPublicCoverage2018UT": 300,
    "ACSPublicCoverage2018VT": 300,
    "ACSPublicCoverage2018VA": 300,
    "ACSPublicCoverage2018WA": 300,
    "ACSPublicCoverage2018WV": 300,
    "ACSPublicCoverage2018WI": 300,
    "ACSPublicCoverage2018WY": 300,
    "ACSPublicCoverage2018PR": 300,
}

dict_arch = {
    "adult": "transformer",
    "aloi": "resnet",
    "bank": "resnet",
    "catsdogs": "vgg",
    "celeba": "resnet50",
    "compass": "transformer",
    "heloc": "transformer",
    "letter": "transformer",
    "house": "transformer",
    "newspaper": "transformer",
    "online": "transformer",
    "ucicredit": "resnet",
    "cifar10": "vgg",
    "covtype": "transformer",
    "food101": "resnet",
    "giveme": "resnet",
    "helena": "resnet",
    "higgs": "transformer",
    "indian": "resnet",
    "jannis": "transformer",
    "mammography": "resnet",
    "miniboone": "resnet",
    "MNIST": "resnet",
    "phishing": "resnet",
    "shuttle": "resnet",
    "stanfordcars": "resnet",
    "SVHN": "vgg",
    "waterbirds": "resnet50",
    "oxfordpets": "resnet",
    "xray": "resnet",
    "eye": "transformer",
    "electricity": "transformer",
    "pol": "transformer",
    "upselling": "transformer",
    "rl": "transformer",
    "magic": "transformer",
    "phoneme": "transformer",
    "kddipums97": "transformer",
    "organamnist": "resnet18",
    "organcmnist": "resnet18",
    "chestmnist": "resnet18",
    "breastmnist": "resnet18",
    "dermamnist": "resnet18",
    "bloodmnist": "resnet18",
    "pneumoniamnist": "resnet18",
    "tissuemnist": "resnet18",
    "octmnist": "resnet18",
    "pathmnist": "resnet18",
    "retinamnist": "resnet18",
    "organsmnist": "resnet18",
    "FashionMNIST": "resnet",
    "cifar100": "vgg",
    "ACSPublicCoverage2018AL": "transformer",
    "ACSPublicCoverage2018AK": "transformer",
    "ACSPublicCoverage2018AZ": "transformer",
    "ACSPublicCoverage2018AR": "transformer",
    "ACSPublicCoverage2018CA": "transformer",
    "ACSPublicCoverage2018CO": "transformer",
    "ACSPublicCoverage2018CT": "transformer",
    "ACSPublicCoverage2018DE": "transformer",
    "ACSPublicCoverage2018FL": "transformer",
    "ACSPublicCoverage2018GA": "transformer",
    "ACSPublicCoverage2018HI": "transformer",
    "ACSPublicCoverage2018ID": "transformer",
    "ACSPublicCoverage2018IL": "transformer",
    "ACSPublicCoverage2018IN": "transformer",
    "ACSPublicCoverage2018IA": "transformer",
    "ACSPublicCoverage2018KS": "transformer",
    "ACSPublicCoverage2018KY": "transformer",
    "ACSPublicCoverage2018LA": "transformer",
    "ACSPublicCoverage2018ME": "transformer",
    "ACSPublicCoverage2018MD": "transformer",
    "ACSPublicCoverage2018MA": "transformer",
    "ACSPublicCoverage2018MI": "transformer",
    "ACSPublicCoverage2018MN": "transformer",
    "ACSPublicCoverage2018MS": "transformer",
    "ACSPublicCoverage2018MO": "transformer",
    "ACSPublicCoverage2018MT": "transformer",
    "ACSPublicCoverage2018NE": "transformer",
    "ACSPublicCoverage2018NV": "transformer",
    "ACSPublicCoverage2018NH": "transformer",
    "ACSPublicCoverage2018NJ": "transformer",
    "ACSPublicCoverage2018NM": "transformer",
    "ACSPublicCoverage2018NY": "transformer",
    "ACSPublicCoverage2018NC": "transformer",
    "ACSPublicCoverage2018ND": "transformer",
    "ACSPublicCoverage2018OH": "transformer",
    "ACSPublicCoverage2018OK": "transformer",
    "ACSPublicCoverage2018OR": "transformer",
    "ACSPublicCoverage2018PA": "transformer",
    "ACSPublicCoverage2018RI": "transformer",
    "ACSPublicCoverage2018SC": "transformer",
    "ACSPublicCoverage2018SD": "transformer",
    "ACSPublicCoverage2018TN": "transformer",
    "ACSPublicCoverage2018TX": "transformer",
    "ACSPublicCoverage2018UT": "transformer",
    "ACSPublicCoverage2018VT": "transformer",
    "ACSPublicCoverage2018VA": "transformer",
    "ACSPublicCoverage2018WA": "transformer",
    "ACSPublicCoverage2018WV": "transformer",
    "ACSPublicCoverage2018WI": "transformer",
    "ACSPublicCoverage2018WY": "transformer",
    "ACSPublicCoverage2018PR": "transformer",
}

image_data = [
    "catsdogs",
    "celeba",
    "cifar10",
    "food101",
    "MNIST",
    "stanfordcars",
    "SVHN",
    "waterbirds",
    "oxfordpets",
    "xray",
    "organamnist",
    "organcmnist",
    "chestmnist",
    "breastmnist",
    "dermamnist",
    "bloodmnist",
    "pneumoniamnist",
    "tissuemnist",
    "octmnist",
    "pathmnist",
    "retinamnist",
    "organsmnist",
    "FashionMNIST",
    "cifar100",
]

dict_batch = {
    "adult": 256,
    "aloi": 512,
    "bank": 128,
    "catsdogs": 128,
    "celeba": 128,
    "cifar10": 128,
    "covtype": 1024,
    "food101": 256,
    "giveme": 512,
    "helena": 512,
    "higgs": 512,
    "indian": 128,
    "jannis": 512,
    "mammography": 128,
    "miniboone": 256,
    "MNIST": 128,
    "phishing": 128,
    "shuttle": 128,
    "stanfordcars": 128,
    "SVHN": 128,
    "waterbirds": 128,
    "oxfordpets": 128,
    "compass": 128,
    "heloc": 128,
    "letter": 128,
    "house": 128,
    "newspaper": 128,
    "online": 128,
    "ucicredit": 128,
    "xray": 128,
    "eye": 128,
    "electricity": 128,
    "pol": 128,
    "upselling": 128,
    "rl": 128,
    "magic": 128,
    "phoneme": 128,
    "kddipums97": 128,
    "organamnist": 256,
    "organcmnist": 128,
    "chestmnist": 512,
    "breastmnist": 64,
    "dermamnist": 128,
    "bloodmnist": 128,
    "pneumoniamnist": 128,
    "tissuemnist": 1024,
    "octmnist": 512,
    "pathmnist": 512,
    "retinamnist": 128,
    "organsmnist": 128,
    "FashionMNIST": 128,
    "cifar100": 128,
    "ACSPublicCoverage2018AL": 128,
    "ACSPublicCoverage2018AK": 128,
    "ACSPublicCoverage2018AZ": 128,
    "ACSPublicCoverage2018AR": 128,
    "ACSPublicCoverage2018CA": 512,
    "ACSPublicCoverage2018CO": 128,
    "ACSPublicCoverage2018CT": 128,
    "ACSPublicCoverage2018DE": 128,
    "ACSPublicCoverage2018FL": 256,
    "ACSPublicCoverage2018GA": 128,
    "ACSPublicCoverage2018HI": 128,
    "ACSPublicCoverage2018ID": 128,
    "ACSPublicCoverage2018IL": 256,
    "ACSPublicCoverage2018IN": 128,
    "ACSPublicCoverage2018IA": 128,
    "ACSPublicCoverage2018KS": 128,
    "ACSPublicCoverage2018KY": 128,
    "ACSPublicCoverage2018LA": 128,
    "ACSPublicCoverage2018ME": 128,
    "ACSPublicCoverage2018MD": 128,
    "ACSPublicCoverage2018MA": 128,
    "ACSPublicCoverage2018MI": 128,
    "ACSPublicCoverage2018MN": 128,
    "ACSPublicCoverage2018MS": 128,
    "ACSPublicCoverage2018MO": 128,
    "ACSPublicCoverage2018MT": 128,
    "ACSPublicCoverage2018NE": 128,
    "ACSPublicCoverage2018NV": 128,
    "ACSPublicCoverage2018NH": 128,
    "ACSPublicCoverage2018NJ": 128,
    "ACSPublicCoverage2018NM": 128,
    "ACSPublicCoverage2018NY": 256,
    "ACSPublicCoverage2018NC": 128,
    "ACSPublicCoverage2018ND": 128,
    "ACSPublicCoverage2018OH": 128,
    "ACSPublicCoverage2018OK": 128,
    "ACSPublicCoverage2018OR": 128,
    "ACSPublicCoverage2018PA": 256,
    "ACSPublicCoverage2018RI": 128,
    "ACSPublicCoverage2018SC": 128,
    "ACSPublicCoverage2018SD": 128,
    "ACSPublicCoverage2018TN": 128,
    "ACSPublicCoverage2018TX": 512,
    "ACSPublicCoverage2018UT": 128,
    "ACSPublicCoverage2018VT": 128,
    "ACSPublicCoverage2018VA": 128,
    "ACSPublicCoverage2018WA": 128,
    "ACSPublicCoverage2018WV": 128,
    "ACSPublicCoverage2018WI": 128,
    "ACSPublicCoverage2018WY": 128,
    "ACSPublicCoverage2018PR": 128,
}
dict_tabular = {
    k: False if k in image_data else True for k in sorted(dict_batch.keys())
}
dict_baseline = {
    "sele": "sele",
    "reg": "reg",
    "cn": "cn",
    "plugin": "plugin",
    "pluginauc": "plugin",
    "scross": "plugin",
    "ensemble": "plugin",
    "ensemble_sr": "plugin",
    "scross_ens": "plugin",
    "aucross": "plugin",
    "selnet": "selnet",
    "selnet_sr": "selnet",
    "selnet_em": "selnet_em",
    "selnet_em_sr": "selnet_em",
    "sat": "sat",
    "sat_sr": "sat",
    "sat_em": "sat_em",
    "sat_em_sr": "sat_em",
    "dg": "dg",
    "selnet_te": "selnet_te",
    "selnet_te_sr": "selnet_te",
    "sat_te": "sat_te",
    "sat_te_sr": "sat_te",
}
