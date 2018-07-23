#!/usr/bin/env python

from time import sleep, strftime, localtime  

fields_required=25
sourcefile = open("/home/frank/IbPy/market_data_raw.csv",'r')
csvfile = open("market_data_"+str(fields_required)+".csv",'wb') 
for line in sourcefile:
    raw=line.split("|")
    symbol = raw[0] 
    data = {k:v for k,v in (x.split('=') for x in raw[1].split(";") if len(x)>0)}
    if len(raw)>=4 and float(raw[2])>0 and float(raw[3])>0:
	#print "parsing data for %s" % symbol
        close = raw[2]
	volatility = raw[3]
	cf_ps = data.get('TTMCFSHR')
	fcf_ps = data.get('TTMFCFSHR')
	bv_ps = data.get('QTANBVPS')
	op_margin = data.get('TTMOPMGN')
	revenue_ps = data.get('TTMREVPS')
	eps_n = data.get('AEPSNORM')
	eps = data.get('TTMEPSXCLX')
	ebitd = data.get('TTMEBITD')
	pretax_margin = data.get('APTMGNPCT')
	cash_ps=data.get('QCSHPS')
	eps_change = data.get('EPSCHNGYR')
	eps_growth = data.get('EPSTRENDGR')
	current = data.get('QCURRATIO')
	lt_de = data.get('QLTD2EQ')
	quick = data.get('QQUICKRATI')
	total_de = data.get('QTOTD2EQ')
	revenue_change=data.get('REVCHNGYR')
	revenue_growth=data.get('REVTRENDGR')
	ebt=data.get('TTMEBT')
	gross_margin=data.get('TTMGROSMGN')
	net_income=data.get('TTMNIAC')
	net_profit_margin=data.get('TTMNPMGN')
	PR1WKPCT = data.get('PR1WKPCT')
        PR4WKPCT = data.get('PR4WKPCT')
        aebit = data.get('AEBIT')
        ANIACNORM = data.get('ANIACNORM') 
        beta = data.get('BETA')
        QFPSS = data.get('QFPSS')
        EV2EBITDA_Cur = data.get('EV2EBITDA_Cur')
        AFPSS = data.get('AFPSS') 
        QFPRD = data.get('QFPRD')
	AEPSXCLXOR = data.get('AEPSXCLXOR')
        TTMREVPERE = data.get('TTMREVPERE')
        TTMROEPCT = data.get('TTMROEPCT')
        TTMINTCOV = data.get('TTMINTCOV')
        MKTCAP = data.get('MKTCAP')
        QEBITDA = data.get('QEBITDA')
	QEBIT = data.get('QEBIT')
        AROIPCT = data.get('AROIPCT')
        PR52WKPCT = data.get('PR52WKPCT')

        
        #print symbol+"|"+close+"|"+PR52WKPCT+"|"+difference
        

	PR13WKPCT = data.get('PR13WKPCT')
	QBVPS = data.get('QBVPS')
	PRYTDPCTR = data.get('PRYTDPCTR')
	NetDebt_I = data.get('NetDebt_I')
	AEBTNORM = data.get('AEBTNORM')
	AREVPS = data.get('AREVPS')
	TTMPTMGN = data.get('TTMPTMGN')
	TTMPRFCFPS = data.get('TTMPRFCFPS')
	TTMNIPEREM = data.get('TTMNIPEREM')
	QSICF = data.get('QSICF')
	TTMPR2REV = data.get('TTMPR2REV')
	QLSTD = data.get('QLSTD')
	QCASH = data.get('QCASH')
	TTMINVTURN = data.get('TTMINVTURN')
	TTMREV = data.get('TTMREV')
	AFPRD = data.get('AFPRD')
	QTL = data.get('QTL')
	QTA = data.get('QTA')
	ASCEX = data.get('ASCEX')
	Frac52Wk = data.get('Frac52Wk')
	ADIV5YAVG = data.get('ADIV5YAVG')
	YIELD = data.get('YIELD')
	TTMEPSCHG = data.get('TTMEPSCHG')
	QOTLO = data.get('QOTLO')
	TTMFCF = data.get('TTMFCF')
	TTMROAPCT = data.get('TTMROAPCT')
	TTMROIPCT = data.get('TTMROIPCT')
	TTMREVCHG = data.get('TTMREVCHG')
        TTMPAYRAT  = data.get('TTMPAYRA')
        

	fields = [symbol,close,volatility,cf_ps,fcf_ps,bv_ps,op_margin,revenue_ps,eps_n,eps,ebitd,pretax_margin,cash_ps,
	     eps_change,eps_growth,current,lt_de,quick,total_de,revenue_change,revenue_growth,ebt,gross_margin,net_income,net_profit_margin,aebit,ANIACNORM,beta,QFPSS,EV2EBITDA_Cur,AFPSS,QFPRD,AEPSXCLXOR,TTMREVPERE,TTMROEPCT,TTMINTCOV,
             MKTCAP,QEBITDA,QEBIT,AROIPCT,QBVPS,PRYTDPCTR,NetDebt_I,AEBTNORM,AREVPS,TTMPTMGN,TTMPRFCFPS,TTMNIPEREM,
             QSICF,TTMPR2REV,QLSTD,QCASH,TTMINVTURN,TTMREV,AFPRD,QTL,QTA,ASCEX,ADIV5YAVG,YIELD,TTMEPSCHG,QOTLO,TTMFCF,TTMROAPCT,
             TTMROIPCT,TTMREVCHG]


        fields_not_none = [x.strip() for x in fields if x != None]

	fields_valid = [x for x in fields_not_none if x !="-99999.99"]
#	print symbol+"|"+str(len(fields_valid))

	if len(fields_valid)>fields_required+1:
		row = ",".join(["" if field == None or field.strip()=="-99999.99"  else field.strip() for field in fields])
		csvfile.write(row+"\n")
	#    else:
	#print "not enough fields for "+symbol


csvfile.close()
