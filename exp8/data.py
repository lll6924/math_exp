def get6():
    with open("data6","r") as f:
        lines=f.readlines()
        data=[]
        for l in lines:
            p=l.split()
            data.append({"height":int(p[0]),"weight":int(p[1])})
    return data

def get10():
    with open("data10","r")as f:
        lines=f.readlines()
        data=[]
        for l in lines:
            p=l.split()
            data.append({"country":p[0],"gender":p[1],"age":int(p[2]),"mean":float(p[3]),"std":float(p[4])})
    return data
