def get5():
    with open("data5","r") as f:
        lines=f.readlines()
        data=[]
        for l in lines:
            p=l.split()
            data.append({"y":float(p[0]),"x1":float(p[1]),"x2":float(p[2]),"x3":float(p[3])})
    return data

def get10():
    with open("data10","r")as f:
        lines=f.readlines()
        data=[]
        for l in lines:
            p=l.split()
            data.append({"y":float(p[0]),"x1":float(p[1]),"x2":float(p[2])})
    return data

def get11():
    with open("data11","r")as f:
        lines=f.readlines()
        data=[]
        for l in lines:
            p=l.split()
            data.append({"y":float(p[0]),"x1":float(p[1]),"x2":float(p[2]),"x3":float(p[3])})
    return data