import pygimli as pg
import glob 

data = pg.load('data_0.data')
datalist = glob.glob('*.ohm')

for d in datalist:
    print('Loading ' + d)
    t = pg.load(d)
    try:
        t("k")
        print('Datafile has geometric factors already. (' + str(d) + ')')
    except:
        t.set("k", data("k"))
    t.save((d[:-4]+'_k'+'.ohm'),"a b m n r err k")
    print('Saved ' + d)
