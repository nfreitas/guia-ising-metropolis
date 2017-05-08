import numpy as np

def autocorr(A,k):

    a = np.mean(A)

    c = 0
    for j in range(len(A)-k):
        c += (A[j]-a)*(A[j+k]-a)

    return c/(len(A)-k)


def energy(S,H):

    X, Y = S.shape

    E = -1*H*np.sum(S)

    for x in range(X):
        for y in range(Y):
            E -= S[x,y]*S[x-1,y]
            E -= S[x,y]*S[x,y-1]

    return E 

def update(S,H,b):

    X, Y = S.shape

    x = np.random.randint(X)
    y = np.random.randint(Y)

    dM = -2*S[x,y]

    dE = -1*H*dM
    HN = S[x-1,y] + S[x,y-1] + S[x,(y+1)%Y] + S[(x+1)%X,y]
    dE += -1*HN*dM

    if dE < 0:
        S[x,y] = -1*S[x,y]
    else:
        r = np.random.rand()
        if r < np.exp(-1*b*dE):
            S[x,y] = -1*S[x,y]
        else:
            dM = 0 
            dE = 0

    return S, dM, dE

def measurement(oldS,H,b,ST):
    
    E = [energy(oldS,H)]
    M = [np.sum(oldS)]
    
    for t in range(ST):
        S, dM, dE = update(oldS,H,b)
        M.append(M[-1]+dM)
        E.append(E[-1]+dE)

        oldS = S

    mE = np.mean(E[::500])
    sE = np.std(E[::500])

    mM = np.mean(M[::500])
    sM = np.std(M[::500])

    return mE,sE,mM,sM

L = 32
H = 0

PT = 100000

S = -1 + 2*np.random.randint(2,size=(L,L))

E = [energy(S,H)]
M = [np.sum(S)]

#for t in range(200000):
#    S, dM, dE = update(S,H,.8)
#    M.append(M[-1]+dM)
#    E.append(E[-1]+dE)

T = np.arange(1.5,3.5,.05)
B = 1/T

ME = []
SE = []
MM = []
SM = []

# pretermalization
for t in range(PT):
    S, dM, dE = update(S,H,B[0])

# measurement
for b in B:
    print(b)

    mE,sE,mM,sM = measurement(S,H,b,20*PT)
    ME.append(mE)
    SE.append(sE)
    MM.append(mM)
    SM.append(sM)
