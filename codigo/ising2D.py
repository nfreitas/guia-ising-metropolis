import numpy as np

def autocorr(A,k):
#    print(k)

    a = np.mean(A)

    c = 0
    for j in range(len(A)-k):
        c += (A[j]-a)*(A[j+k]-a)

    return c/(len(A)-k)


def corr_func(S):

    X, Y = S.shape
    C = 0 

    for l in range(X):
        c = [autocorr(S[l,:],k) for k in range(Y-1)]
        C += np.array(c)

    for l in range(Y):
        c = [autocorr(S[:,l],k) for k in range(X-1)]
        C += np.array(c)

    return C/(X+Y-2)

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
    
    C = 0
    count = 0

    for t in range(ST):
        S, dM, dE = update(oldS,H,b)
        M.append(M[-1]+dM)
        E.append(E[-1]+dE)
        oldS = S
        if t%1000 == 0:
            C += corr_func(S)
            count += 1

    mE = np.mean(E[::10])
    sE = np.std(E[::10])

    mM = np.mean(M[::10])
    sM = np.std(M[::10])

    return mE,sE,mM,sM,C/count

L = 32
H = 0

PT = 800000

S = -1 + 2*np.random.randint(2,size=(L,L))
#S = np.ones((L,L))

E = [energy(S,H)]
M = [np.sum(S)]

#for t in range(200000):
#    S, dM, dE = update(S,H,1/2)
#    M.append(M[-1]+dM)
#    E.append(E[-1]+dE)

T = np.arange(1.5,3,.1)
T = T[::-1]
B = 1/T

ME = []
SE = []
MM = []
SM = []
C = []

# pretermalization
for t in range(PT):
    S, dM, dE = update(S,H,B[0])

# measurement
for b in B:
    print(b)

    mE,sE,mM,sM,c = measurement(S,H,b,PT)
    ME.append(mE)
    SE.append(sE)
    MM.append(mM)
    SM.append(sM)
    C.append(c)
