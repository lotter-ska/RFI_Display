import gspread
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



from oauth2client.client import SignedJwtAssertionCredentials

json_key = json.load(open('lkock-ska-d7fb00bcce75.json'))
scope = ['https://spreadsheets.google.com/feeds']

credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)

gc = gspread.authorize(credentials)

sh = gc.open("RFI Spectrum Database")
worksheet = sh.get_worksheet(0)


Fstart=worksheet.col_values(8)
Fstart=Fstart[2:-1]
Fstart=filter(None,Fstart)
Fstart=np.array(Fstart)
Fstart = Fstart.astype(np.float)

Fstop=worksheet.col_values(9)
Fstop=Fstop[2:len(Fstart)+2]
Fstop=np.array(Fstop)

Info=worksheet.col_values(12)
Info=Info[2:len(Fstart)+2]
Info=np.array(Info)

Info_cul=worksheet.col_values(13)
Info_cul=Info_cul[2:len(Fstart)+2]
Info_cul=np.array(Info_cul)
Info_cul=filter(None,Info_cul)


#bandstart=Fstart[Fstop!='']
bandstop=Fstop[Fstop!='']
bandstop = bandstop.astype(np.float)
bandi=Info[Fstop!='']


cul=Fstart[Fstop=='']


np.save('/home/reverb-chamber/Desktop/Ratty/Cul_db.npy',[cul,Info_cul])


