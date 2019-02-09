from random import randint
dict={0:'A',1:'C',2:'T',3:'G'}
def rnd_ATCG():
    r=randint(0,3)
    return dict[r]
def protein(origin,length):
    for i in range(length):
        origin=origin+rnd_ATCG()
    return origin

f=open('StringSearch_Input.txt','w')
f.write(protein('',640000000))
f.close()
#print protein('',50)
