import math
import numpy as np
import timeit
import su2

#import unitarize
#THE R VECTOR IS THE RANDOMLY GENERATED OCTET
#THE S VECTOR IS THE R VECTOR MAPPED TO SU(3)

prod = np.dot
tol = 1e-13 #work on this

#initialize 8 gellmann matrices
#Here I am using 2T_i = λ_i for... reasons
T = np.zeros((8,3,3), dtype=complex)
#λ1
T[0][0][1] = 1./2
T[0][1][0] = 1./2
#λ2
T[1][0][1] = 0. - 1.j/2
T[1][1][0] = 0. + 1.j/2
#λ3
T[2][0][0] = 1./2
T[2][1][1] = -1./2
#λ4
T[3][0][2] = 1./2
T[3][2][0] = 1./2
#λ5
T[4][0][2] = 0. - 1.j/2
T[4][2][0] = 0. + 1.j/2
#λ6
T[5][1][2] = 1./2
T[5][2][1] = 1./2
#λ7
T[6][1][2] = 0. - 1.j/2
T[6][2][1] = 0. + 1.j/2
#λ8
T[7][0][0] = 1./(math.sqrt(3)*2)
T[7][1][1] = 1./(math.sqrt(3)*2)
T[7][2][2] = -1./math.sqrt(3)

def trace(A):
    #THIS FUNCTION IS FOR MATRICES, NOT SU3 VECTORS
    return np.trace(A)

def commutator(a,b):
    comm = np.dot(a,b) - np.dot(b,a)
    return comm

def aCommutator(a,b):
    aComm = np.dot(a,b) + np.dot(b,a)
    return aComm

#INITIALIZE THE F and D TENSORS
f = np.zeros((8,8,8), dtype=complex)
d = np.zeros((8,8,8), dtype=complex)
#D and F TENSORS
for a in range(8):
    for b in range(8):
        for c in range(8):
            f[a][b][c] = -2j*trace(np.dot(np.dot(T[a],T[b]),T[c])-np.dot(np.dot(T[b],T[a]),T[c]))
            d[a][b][c] = 2*trace(np.dot(np.dot(T[a],T[b]),T[c])+np.dot(np.dot(T[b],T[a]),T[c]))

def star(i, R):
    #computes the (a*a)i quantity in the MacFarland paper
    #AN INPUT OF i=1 COMPUTES (a*a)i = dijkajak
    #AN INPUT OF i=2 COMPUTES (a*a)j
    #AN INPUT OF i=3 COMPUTES (a*a)k
    sum = 0.
    star = np.zeros(len(R),dtype=complex)
    if i == 1:
        for a in range(len(R)):
            for b in range(len(R)):
                for c in range(len(R)):
                    sum += R[b]*R[c]*d[a][b][c]
            star[a] = sum
            sum = 0.
    elif i == 2:
        for b in range(len(R)):
            for a in range(len(R)):
                for c in range(len(R)):
                    sum += R[a]*R[c]*d[a][b][c]
            star[b] = sum
            sum = 0.
    elif i == 3:
        for c in range(len(R)):
            for a in range(len(R)):
                for b in range(len(R)):
                    sum += R[a]*R[b]*d[a][b][c]
            star[c] = sum
            sum = 0.
    else:
        print('ERROR, aStar(i,R) requires integer input i = 1,2,3')
    return star


def expMap(R):
    #the map from the R vector to the S vector via the Macfarlane exponential solution
    #the R+1 term shows up because there is an additional term, u0, in the su3 representation
    #THIS FUNCTION DOES NOT WORK AS OF NOW, USE CAYLEYMAP
    u = np.zeros(len(R)+1, dtype=complex)
    tol = 1e-8

    I2 = 0 + 0j
    for i in range(len(R)):
        I2 += R[i]*R[i]
    
    I3 = 0 + 0j
    Rstar = np.zeros(len(R),dtype=complex)
    sum = 0.
    for c in range(len(R)):
        for b in range(len(R)):
            for a in range(len(R)):
                I3 += R[a]*R[b]*R[c]*d[a][b][c]
                sum += R[a]*R[b]*d[a][b][c]
        Rstar[c] = sum
        sum = 0.
    print(Rstar)
    print(I2)
    print(I3)

    xi = np.arccos(np.sqrt(3)*I3*pow(I2,-3/2))
    phi = [0.]*3
    for a in range(3):
        phi[a] = 2*pow(I2/3,1/2)*np.cos(1/3*(xi+2*math.pi*(a+1)))
    
    #we can now solve for u0,uk
    u[0] = 0.
    for a in range(3):
        u[0] += np.cos(phi[a]) + 1j*np.sin(phi[a])
    u[0] *= 1./3.

    x=0
    y=0
    for a in range(3):
        x += -phi[a]*np.exp(phi[a]*1j)/(3*phi[a]*phi[a]-I2)
        y += -np.exp(phi[a]*1j)/(3*phi[a]*phi[a]-I2)

    for k in range(len(R)):
        asum = 0.
        for a in range(3):
            asum += (np.cos(phi[a])+1j*np.sin(phi[a]))*2*(phi[a]*R[k]+Rstar[k])/(3*phi[a]*phi[a]-I2)
        u[k+1] = -1/2*1j*(asum)
    return u


def cayleyMap(b):
    #maps the b (R) vector to the S vector according to the Cayley representation. 
    bstar = np.zeros(8)
    bstar =  star(3,b)
    I2 = 0.    
    for i in range(len(b)):
        I2 += b[i]*b[i]
    I3 = 0.
    for i in range(len(b)):
        for j in range(len(b)):
            for k in range(len(b)):
                I3 += b[i]*b[j]*b[k]*d[i][j][k]
    #useful constant
    cons1 = np.sqrt(4*pow(-9-3*I2,3)+324*I3*I3)
    #defines b0 according to the constraint detU = 1
    b0 = (pow(2,1/3)*(-9-3*I2)/(3*pow(18*I3+cons1,1/3)))-pow(18*I3+cons1,1/3)/(3*pow(2,1/3))

    Omega = 1 + I2 - 3*b0*b0
    u = np.zeros(9,dtype=complex)
    u[0] = 4/3*(1-3*b0*1j)/Omega - 1/3
    for k in range(8):
        u[k+1] = 2j*((1-b0*1j)*b[k]+ bstar[k]*1j)/Omega    
    return(u)            


def multi(u1,u2):
    #SU3 vector multiplication
    #unfortunately, this requires a lot more computation than standard 3x3 matrix multiplication as it stands.
    u3 = np.zeros(9, dtype=complex)
    u3[0] = u1[0]*u2[0]
    for m in range(8):
        u3[0] += 2/3*u1[m+1]*u2[m+1]
        for j in range(8):
            for k in range(8):
                u3[m+1] += (d[j][k][m]+1j*f[j][k][m])*(u1[j+1]*u2[k+1])
        u3[m+1] += u1[0]*u2[m+1] + u2[0]*u1[m+1]

    return u3


def det(u):
    #determinant of matrix where u is the S vector
    #wrote it like this just so you could see the whole expression without scrolling far to the right
    determinant = u[0]**3 + u[3]*u[4]**2 + u[3]*u[5]**2 - u[0]*(np.dot(u,u) - u[0]*u[0])
    determinant += 2*u[1]*u[4]*u[6] + 2*u[2]*u[5]*u[6] - u[3]*u[6]**2 - 2*u[2]*u[4]*u[7] + 2*u[1]*u[5]*u[7]
    determinant += -u[3]*u[7]**2 + u[8]*(2*u[1]**2 + 2*u[2]**2 + 2*u[3]**2 - u[4]**2 - u[5]**2 - u[6]**2 - u[7]**2 - 2*u[8]**2/3)/(math.sqrt(3))   
    return determinant

def tr(u):
    #TRACE FUNCTION FOR SU3 VECTOR
    return 3*u[0]

def dagger(U):
    #hermitian conjugate
    #RIGHT NOW THIS FUNCTION DOES NOT HELP IF REPRESENTED AS u0 + i*uk*lambdak
    Udag = np.zeros(9,dtype=complex)
    for i in range(len(U)):
        Udag[i] = np.conj(U[i])
    return Udag

def vol(La):
    product = 1
    for x in range(len(La)):
        product *= La[x]
    return product

#dimensions of the array in a dictionary
def dim(La):
    D = {}
    for x in range(len(La)):
        D.update({x:La[x]})
    return D


def p2i(point, La):
    #from a point to an index
    #basically su(2)
    #point: [x,y,z,t] format
    #La: array-like; each element describes the length of the lattice dimension

    return ((La[2] * La[1] * La[0] * point[3]) + (La[1] * La[0] * point[2]) + (La[0] * point[1]) + (point[0]))


def i2p(ind,La):
    #from an index to a point
    #same comment as p2i
    #ind: index as int
    #La: array-like, same old

    v = La[0] * La[1] * La[2]
    a = La[0] * La[1]
    l = La[0]
    t = divmod(ind, v)
    z = divmod(t[1], a)
    y = divmod(z[1], l)
    x = divmod(y[1], 1)

    return np.array((x[0], y[0], z[0], t[0]))


def parity(pt):
    #parity of point on the lattice ; remainder==0, odd parity; ==1, even
    return np.sum(pt)%2

def hstart():
    #hot start on the lattice
    R = np.zeros(8)
    for i in range(len(R)):
        R[i] = np.random.uniform(-1,1)
    U = cayleyMap(R)
    return U

#def update(U):
    #sean's : make a random su(2) matrix near the identity
    #mine : didn't we just put in unitarize?
#    return unitarize.reunitarize(U)

def reunitarize(U):

     return U


def update(U):
    return reunitarize(multi(U, hstart()))


def cstart():
    # 3x3 identity matrix
    U = np.zeros(9,dtype=complex)
    U[0] = 1.
    return(U)

def mstart():
    mrand = np.random.random()
    if(mrand >= 0.5):
        U = hstart()
    else:
        U = cstart()
    return U


def mupi(ind, mu, La):
    #increment a position in the mu-th direction
    #ind: index
    #mu: the direction of incrementation
    #La: array-like; elements describe length of lattice [x,y,z,t]

    pos = i2p(ind,La)
    if (pos[mu] + 1 >= La[mu]):
        pos[mu] = 0
    else:
        pos[mu] += 1
    return p2i(pos,La)

#apparently Sean doesn't use this function in PvB
def getMups(vol, numdim, La):
    #gets mups array
    #vol: volume/points on lattice
    #numdim: # dimensions on lattice
    #La: array-like

    mups = np.zeros((vol,numdim), int)
    for i in range(0, vol):
        for mu in range(0, numdim):
            mups[i,mu] = mupi(i,mu,La)

    return mups

def mudowni(ind, mu, La):
    #decrment a position in the mu-th direction
    #ind: index
    #mu: the direction of incremenatation
    #La: array-like; elements describe length of lattice [x,y,z,t]

    point = i2p(ind, La)
    if (point [mu] - 1 < 0):
        point[mu] = La[mu] - 1
    else:
        point[mu] -= 1
    return p2i(point, La)

def plaq(U, U0i, mups, mu, nu):
    #compute the plaquette ;; moving in forward dir only
    #copy is important bc pointers in python
    #U: La containing gauge fields for every point
    #U0i: int lattice pt index of starting point
    #mups: mups array; incrementing in muth dir from U0i
    #mu: int = one of pts on lattice, 0:x, 1:y, 2:z, 3:t
    #nu: int corresponding to another dir on lattice *see above*

    U0 = U[U0i][mu].copy()
    U1 = U[mups[U0i,mu]][nu].copy()
    U2 = dagger(U[mups[U0i,nu]][mu].copy())
    U3 = dagger(U[U0i][nu].copy())

#    prod1 = np.matmul(U0,U1)
#    prod2 = np.matmul(U2,U3)
#    prod3 = np.matmul(prod1,prod2)

#    return trace(prod3)
    return tr(multi(multi(U0,U1),multi(U2,U3))).real


def link(U,U0i,mups,mu):
     #trace of link btwn 2 points
     #copy is important bc pointers in python
     #U: La containing gauge fields for every point
     #U0i: int lattice pt index of starting point
     #mups: mups array; incrementing in muth dir from U0i
     #mu: int = one of pts on lattice, 0:x, 1:y, 2:z, 3:t
     #nu: int corresponding to another dir on lattice *see above*

     U0 = U[U0i][mu].copy()

     return tr(U0)

def getstaple(U,U0i,mups,mudowns,mu):
    #returns value of staple
    #be careful about mups and mudowns, don't want it to be too close to the function names, but I prefer uniformity
    #U: La containing gauge fields for every point
    #U0i: int lattice pt index of starting point
    #mups: mups array; incrementing in muth dir from U0i in forward dir
    #mudowns: mups array; decrementing in muth dir from U0i in backward dir
    #mu: index corresponding to a direction 0:x, 1:y, 2:z, 3:t

    value = 0.0
    mm = list(range(4))
    mm = [i for i in range(4)]
    mm.remove(mu)
    for nu in mm:
        #if nu != mu:
        #forward staple
        value += staple(U, U0i, mups, mudowns, mu, nu, 1)

        #reverse staple
        value += staple(U, U0i, mups, mudowns, mu, nu, -1)
    return value

def staple(U,U0i,mups,mudowns,mu,nu,signnu):
    #compute staple in the mu-nu plane

    #U: La containing gauge fields for every point
    #U0i: int lattice pt index of starting point
    #mups: mups array; incrementing in muth dir from U0i in forward dir
    #mudowns: mups array; decrementing in muth dir from U0i in backward dir
    #mu: index corresponding to a direction 0:x, 1:y, 2:z, 3:t
    #nu: ^^^ but it's a new direction and must be different
    #signnu: either 1 or -1, corresponding to forward or backward respectively

    #forward
    if (signnu == 1):
        U1 = U[mups[U0i, mu]][nu].copy()
        U2 = dagger(U[mups[U0i, nu]][mu].copy())
        U3 = dagger(U[U0i][nu].copy())
    #backward
    else:
        U1 = dagger(U[mudowns[mups[U0i, mu],nu]][nu].copy())
        U2 = dagger(U[mudowns[U0i, nu]][mu].copy())
        U3 = U[mudowns[U0i, nu]][nu].copy()

    return multi(multi(U1,U2),U3)

def calcPlaq(U,vol,mups):
    #calc avg value of plaquettes about all lattice points
    #U: La containing gauge fields for every point
    #vol: int volume of lattice = # of points on lattice
    #mups: mups array

    plaquettes = np.zeros(6*vol) #6 directions on each point
    j = 0
    for i in range(vol):
        for mu in range(4):
            for nu in range(mu+1,4):
                plaquettes[j] = 1.0 - plaq(U,i,mups,mu,nu)/3.0
                j = j+1
    avgPlaq = np.mean(plaquettes)

    return avgPlaq

def calcU_i(U,vol,D,mups):
    #calculates the average values of spacial links in lattice
    #U: La containing gauge fields for every point
    #vol: int volume of lattice = # of points on lattice
    #D: not entirely sure? for rn, assuming it's the D returned in dim(La), which is the dimensions of the array in a dictionary
    #new theory on D (talk to Sean about later); D=dimensions, bc putting in La works and gives a result -- but then why wouldn't you just call it La? idk it might be wrong, but it's better than the errors I was getting before
    #mups: mups array
    #as per Sean: D is from when I had dumbbrain and had weird ideas about how things would work. D should be a dictionary from dim(La).
    #BUT dim(La) didn't quite work for me, but since Sean said this never really came up after a point in the pvb code so maybe this isn't the end of the world? Idk I'll keep tinkering at it and come back to it

    spaceLink = np.zeros((3*vol), dtype=complex)
    j = 0
    for i in range(vol):
        for mu in range(D-1):
            #spaceLink[j] = link(U,i,mups,mu)
            spaceLink[j] = link(U,i,mups,mu)
            j = j+1
    U_i = np.mean(spaceLink)
    return U_i/2.

def calcU_t(U,vol,mups):
    #calculates the average values of time links in lattice
    #U: La containing gauge fields for every point
    #vol: int volume of lattice = # of points on lattice
    #mups: mups array

    timeLink = np.zeros((vol), dtype=complex)
    i = 0
    for i in range(vol):
        timeLink[i] = link(U,i,mups,3)

    U_t = np.mean(timeLink)
    return U_t/2.

def hbath(k, r, beta):
    #creates an SU(2) matrix via the heat bath method (see Creutz)
    #the matrix is written in real valued form
    a = np.exp(-2*beta*k)
    b = 1
    boolean = False
    while boolean is False:	
        x = np.random.uniform(a,b)
        a0 = 1. + np.log(x)/(beta*k)
        reject = 1 - np.sqrt(1-a0*a0)

        if(1 - a0*a0 <= 0):
            print('BOUND ERROR')
            return np.zeros(4)

        newrand = np.random.rand()
        if newrand > reject:
            boolean = True	

    absa = math.sqrt(1 - a0*a0)
    theta = np.random.uniform(0,math.pi)
    phi = np.random.uniform(0, 2*math.pi)
    a1 = absa*np.cos(phi)*np.sin(theta)
    a2 = absa*np.sin(phi)*np.sin(theta)
    a3 = absa*np.cos(theta)
    Un = np.array([a0, a1, a2, a3])
    Ubar = r/k
    Uinv = su2.dag(Ubar)
    U = su2.mult(Un,Uinv)

    return U

def stapleleft(staples):
    #returns the real valued form for the upper left 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    r = np.zeros(4)
    r[0] = np.real(staples[0]+staples[8]/math.sqrt(3))
    r[1] = np.imag(staples[1])
    r[2] = np.real(-1j*staples[2])
    r[3] = np.imag(staples[3])
    return r

def stapleright(staples):
    #returns the real valued form for the bottom right 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    r = np.zeros(4)
    r[0] = 0.5*np.real(2*staples[0]-staples[3]-staples[8]/math.sqrt(3))
    r[1] = np.imag(staples[6])
    r[2] = np.real(-1j*staples[7])
    r[3] = 0.5*np.imag(-staples[3]+3*staples[8]/math.sqrt(3))
    return r

def staple_k(staples,k):
    #returns the real valued form for the bottom right 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    r = np.zeros(4)
    r[0] = 0.5*(staples[k][k].real + staples[k+1][k+1].real)
    r[1] = 0.5*(staples[k][k+1].imag + staples[k+1][k].imag)
    r[2] = 0.5*(staples[k][k+1].real - staples[k+1][k].real)
    r[3] = 0.5*(staples[k][k].imag - staples[k+1][k+1].imag)
    return r

def stapleleft1(staples):
    #returns the real valued form for the upper left 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    # r = np.zeros(4)
    # r[0] = 0.5*(staples[0][0].real + staples[1][1].real)
    # r[1] = 0.5*(staples[0][1].imag + staples[1][0].imag)
    # r[2] = 0.5*(staples[0][1].real - staples[1][0].real)
    # r[3] = 0.5*(staples[0][0].imag - staples[1][1].imag)
    return staple_k1(staples,1)#r

def stapleright1(staples):
    #returns the real valued form for the bottom right 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    # r = np.zeros(4)
    # r[0] = 0.5*(staples[1][1].real + staples[2][2].real)
    # r[1] = 0.5*(staples[1][2].imag + staples[2][1].imag)
    # r[2] = 0.5*(staples[1][2].real - staples[2][1].real)
    # r[3] = 0.5*(staples[1][1].imag - staples[2][2].imag)
    return staple_k1(staples,1)#r

def staple_k1(staples,k):
    #returns the real valued form for the bottom right 2x2 submatrix of the staples
    #this process is necessary for the probability distribution
    r = np.zeros(4)
    r[0] = 0.5*(staples[k][k].real + staples[k+1][k+1].real)
    r[1] = 0.5*(staples[k][k+1].real + staples[k+1][k].real)
    r[2] = 0.5*(staples[k][k+1].real - staples[k+1][k].real)
    r[3] = 0.5*(staples[k][k].real - staples[k+1][k+1].real)
    return r


def rightblock(a):
    #converts an SU2 lower right sub matrix to an SU3 vector
    #the SU2 input is written in real valued form
    u = np.zeros(9,dtype=complex)
    u[0] = (1+2*a[0])/3.
    u[3] = (1-a[0]-1j*a[3])/2.
    u[6] = 1j*a[1]
    u[7] = 1j*a[2]
    u[8] = math.sqrt(3)*(1-a[0]+3j*a[3])/6.
    return u

def leftblock(a):
    #converts an SU2 upper left sub matrix to an SU3 vector
    #the SU2 input is written in real valued form
    u = np.zeros(9,dtype=complex)
    u[0] = (1+2*a[0])/3.
    u[1] = 1j*a[1]
    u[2] = 1j*a[2]
    u[3] = 1j*a[3]
    u[8] = math.sqrt(3)*(-1+a[0])/3.
    return u
