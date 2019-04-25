#%% Convert to res2dinv General
def bert_to_res2d(dataIn = None):
    import numpy as np
    from tkinter import filedialog, Tk
    
    if dataIn is None:
        # Load Data
        Tk().withdraw()
        dataIN = filedialog.askopenfilename(filetypes = (("ohm file","*.ohm"),("all files","*.*")))
    else:
        print('Loading '+ dataIN)
        with open(dataIN, 'r+', newline = '\r\n' ) as f:
            datLine = f.read()
        datLine = datLine.splitlines()
        
        # Get Sensors
        x = []
        y = []
        z = []
        
        nSensors = int(datLine[0])
        print(str(nSensors) +' sensors found')
        for lines in np.arange(nSensors):
            xi = float(datLine[2+lines].split()[0])
            yi = float(datLine[2+lines].split()[1])
            zi = float(datLine[2+lines].split()[2])
            
            x = np.append(x,xi)
            y = np.append(y,yi)
            z = np.append(z,zi)
        
        # Get Data
        nData = int(datLine[nSensors+2])
        print(str(nData) +' data found')
        headers = datLine[nSensors+3].split()
        datMat = np.zeros((nData,len(headers)-1))
        
        a = []
        b = []
        m = []
        n = []
        r = []
        rhoa = []
        
        for line in datLine[(nSensors+4):]:
        #    print(line)
            datMatI = line.split()
            if len(datMatI) == len(headers)-1:
                ai = int(datMatI[0])
                a = np.append(a,ai)
                bi = int(datMatI[1])
                b = np.append(b,bi)
                mi = int(datMatI[2])
                m = np.append(m,mi)
                ni = int(datMatI[3])
                n = np.append(n,ni)
                if 'rhoa' in headers:
                    rhoai = float(datMatI[headers.index('rhoa')-1])
                    rhoa = np.append(rhoa,rhoai)
                if 'r' in headers:
                    ri = float(datMatI[headers.index('r')-1])
                    r = np.append(r,ri)
        sensDx = x[1]-x[0]
        
        # Assemble RES2DINV Structures
        with open(dataIN[:-4] + "_res2dinv.dat", "w") as dFile:
            dFile.write(dataIN + '\n')
            dFile.write(str(sensDx)+' \n')
            dFile.write('11 \n')
            dFile.write('0 \n')
            dFile.write('Type of measurement \n')
            dFile.write('1 \n')
            dFile.write((str(nData)+' \n'))
            dFile.write('2 \n')
            dFile.write('0 \n')
            for i in np.arange(nData):
                ax = x[int(a[i])-1]
                ay = y[int(a[i])-1]
                bx = x[int(b[i])-1]
                by = y[int(b[i])-1]
                mx = x[int(m[i])-1]
                my = y[int(m[i])-1]
                nx = x[int(n[i])-1]
                ny = y[int(n[i])-1]
                r_ = r[i]
                
                fmt = '{:d}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.4E}\n'
                outStringFmt = fmt.format(4, ax, ay, bx, by, mx, my, nx, ny, r_)
                dFile.write(outStringFmt)
        print('Done')
    return None