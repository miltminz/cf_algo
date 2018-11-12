class Algorithms:
    @staticmethod
    def brun():
        P = [[(1,0,0),(1,1,0),(1,1,1)], [(1,0,0),(1,0,1),(1,1,1)], [(0,0,1),(1,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,1,1)], [(0,1,0),(0,1,1),(1,1,1)], [(0,1,0),(1,1,0),(1,1,1)]]
        M = [matrix(QQ,[[1,1,0],[0,1,0],[0,0,1]]), matrix(QQ,[[1,0,1],[0,1,0],[0,0,1]]), matrix(QQ,[[1,0,0],[0,1,0],[1,0,1]]), matrix(QQ,[[1,0,0],[0,1,0],[0,1,1]]),  matrix(QQ,[[1,0,0],[0,1,1],[0,0,1]]), matrix(QQ,[[1,0,0],[1,1,0],[0,0,1]])]
        return CFAlgo2(P,M)
    
    @staticmethod
    def selmer():
        P = [[(2,1,1),(1,1,0),(1,1,1)], [(2,1,1),(1,0,1),(1,1,1)], [(1,1,2),(1,0,1),(1,1,1)], [(1,1,2),(0,1,1),(1,1,1)], [(1,2,1),(0,1,1),(1,1,1)], [(1,2,1),(1,1,0),(1,1,1)]]
        M = [matrix(ZZ,[[1,0,1],[0,1,0],[0,0,1]]),matrix(ZZ,[[1,1,0],[0,1,0],[0,0,1]]),matrix(ZZ,[[1,0,0],[0,1,0],[0,1,1]]),matrix(ZZ,[[1,0,0],[0,1,0],[1,0,1]]),matrix(ZZ,[[1,0,0],[1,1,0],[0,0,1]]),matrix(ZZ,[[1,0,0],[0,1,1],[0,0,1]])]
        return CFAlgo2(P,M)
    
    @staticmethod
    def brun_fullysub():
        P = [[(1,0,0),(1,1,0),(1,1,1)], [(1,0,0),(1,0,1),(1,1,1)], [(0,0,1),(1,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,1,1)], [(0,1,0),(0,1,1),(1,1,1)], [(0,1,0),(1,1,0),(1,1,1)]]
        M = [matrix(ZZ,[[1,1,0],[0,1,0],[0,0,1]]), matrix(ZZ,[[1,-1,0],[0,1,0],[0,-1,1]]).inverse(), matrix(ZZ,[[1,0,0],[0,1,0],[1,0,1]]), matrix(ZZ,[[1,0,0],[-1,1,0],[-1,0,1]]).inverse(),matrix(ZZ,[[1,0,0],[0,1,1],[0,0,1]]), matrix(ZZ,[[1,0,-1],[0,1,-1],[0,0,1]]).inverse()]
        return CFAlgo2(P,M)
   
    
    @staticmethod
    def cassaigne():
        P = [[(1,0,0),(0,1,0),(1,0,1)],[(0,0,1),(0,1,0),(1,0,1)]]
        M = [matrix(ZZ,[[1,0,-1],[0,0,1],[0,1,0]]).inverse(),matrix(ZZ,[[0,1,0],[1,0,0],[-1,0,1]]).inverse()]
        return CFAlgo2(P,M)
   

    @staticmethod
    def brun_fullysub1():
        P = [[(1,0,0),(1,1,0),(1,1,1)], [(1,0,0),(1,0,1),(1,1,1)], [(0,0,1),(1,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,1,1)], [(0,1,0),(0,1,1),(1,1,1)], [(0,1,0),(1,1,0),(1,1,1)]]
        M = [matrix(ZZ,[[1,1,0],[0,1,0],[0,0,1]]), matrix(ZZ,[[1,-1,0],[0,1,0],[0,-1,1]]).inverse(), matrix(ZZ,[[1,-1,0],[0,1,0],[0,-1,1]]).inverse(), matrix(ZZ,[[1,0,0],[0,1,0],[0,1,1]]), matrix(ZZ,[[1,0,0],[0,1,1],[0,0,1]]), matrix(QQ,[[1,0,0],[1,1,0],[0,0,1]])]
        return CFAlgo2(P,M)
    
    @staticmethod
    def brun_fullysub2():
        P = [[(1,0,0),(1,1,0),(1,1,1)], [(1,0,0),(1,0,1),(1,1,1)], [(0,0,1),(1,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,1,1)], [(0,1,0),(0,1,1),(1,1,1)], [(0,1,0),(1,1,0),(1,1,1)]]
        M = [matrix(ZZ,[[1,1,0],[0,1,0],[0,0,1]]), matrix(ZZ,[[1,-1,0],[0,1,0],[0,-1,1]]).inverse(), matrix(ZZ,[[1,-1,0],[0,1,0],[0,-1,1]]).inverse(), matrix(ZZ,[[1,0,0],[-1,1,0],[-1,0,1]]).inverse(), matrix(ZZ,[[1,0,0],[-1,1,0],[-1,0,1]]).inverse(), matrix(QQ,[[1,0,0],[1,1,0],[0,0,1]])]
        return CFAlgo2(P,M)
     
    @staticmethod
    def brun_sorted():
        P = [[(1,0,0),(2,1,0),(1,1,1)], [(2,1,0),(2,1,1),(1,1,0)], [(2,1,1),(1,1,0),(1,1,1)]]
        M = [matrix(QQ,[[1,1,0],[0,1,0],[0,0,1]]), matrix(QQ,[[1,1,0],[1,0,0],[0,0,1]]), matrix(QQ,[[1,0,1],[1,0,0],[0,1,0]])]
        return CFAlgo2(P,M)

    @staticmethod
    def poincare():
        P = [[(1,0,0),(1,1,0),(1,1,1)], [(1,0,0),(1,0,1),(1,1,1)], [(0,0,1),(1,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,1,1)], [(0,1,0),(0,1,1),(1,1,1)], [(0,1,0),(1,1,0),(1,1,1)]]
        M = [matrix(QQ,[[1,1,1],[0,1,1],[0,0,1]]), matrix(QQ,[[1,1,1],[0,1,0],[0,1,1]]), matrix(QQ,[[1,1,0],[0,1,0],[1,1,1]]), matrix(QQ,[[1,0,0],[1,1,0],[1,1,1]]),  matrix(QQ,[[1,0,0],[1,1,1],[1,0,1]]), matrix(QQ,[[1,0,1],[1,1,1],[0,0,1]])]
        return CFAlgo2(P,M)

    @staticmethod
    def poincare_sorted():
        P = [[(1, 0, 0), (2, 1, 0), (3, 2, 1)],[(1, 0, 0), (2, 1, 1), (3, 2, 1)],[(1, 1, 1), (2, 1, 1), (3, 2, 1)],[(1, 1, 1), (2, 2, 1), (3, 2, 1)],[(1, 1, 0), (2, 2, 1), (3, 2, 1)],[(1, 1, 0), (2, 1, 0), (3, 2, 1)]]
        M = [matrix(QQ,[[1,1,1],[0,1,1],[0,0,1]]),matrix(3,[1,-1,0,0,0,1,0,1,-1]).inverse(),matrix(3,[0,0,1,1,-1,0,0,1,-1]).inverse(),matrix(3,[0,0,1,0,1,-1,1,-1,0]).inverse(),matrix(3,[0,1,-1,0,0,1,1,-1,0]).inverse(),matrix(3,[0,1,-1,1,-1,0,0,0,1]).inverse()]
        return CFAlgo2(P,M)

    @staticmethod
    def fullysubtractive():
        P = [[(1,0,0),(1,1,1),(0,0,1)], [(0,0,1),(0,1,0),(1,1,1)], [(0,1,0),(1,0,0),(1,1,1)]]
        M = [matrix(QQ,[[1,1,0],[0,1,0],[0,1,1]]), matrix(QQ,[[1,0,0],[1,1,0],[1,0,1]]), matrix(QQ,[[1,0,1],[0,1,1],[0,0,1]])]
        return CFAlgo2(P,M)
    
    @staticmethod
    def arnouxrauzy():
        P = [[(1,0,0),(1,0,1),(1,1,0)], [(0,0,1),(1,0,1),(0,1,1)], [(0,1,0),(1,1,0),(0,1,1)]]
        M = [matrix(QQ,[[1,1,1],[0,1,0],[0,0,1]]), matrix(QQ,[[1,0,0],[0,1,0],[1,1,1]]), matrix(QQ,[[1,0,0],[1,1,1],[0,0,1]])]
        return CFAlgo2(P,M)
    
    # NEW ALGO N0.1 (left -> full+rotation clockwise, right -> full+rotation counterclockwise)
    @staticmethod
    def newalgo1():
        P = [[(1, 0, 0), (1, 1, 0), (0, 0, 1)],[(0, 1, 0), (1, 1, 0), (0, 0, 1)]]
        M = [matrix(QQ,[[0,1,0],[0,0,1],[1,-1,0]]).inverse(),matrix(QQ,[[0,0,1],[1,0,0],[-1,1,0]]).inverse()]
        return CFAlgo2(P,M)
    
    #NEW ALGO NO.1 MODIFIED (left -> full+rotation counterclockwise, right -> full+rotation clockwise)
    @staticmethod
    def newalgo1b():
        P = [[(1, 0, 0), (1, 1, 0), (0, 0, 1)],[(0, 1, 0), (1, 1, 0), (0, 0, 1)]]
        M = [matrix(QQ,[[0,0,1],[1,-1,0],[0,1,0]]).inverse(),matrix(QQ,[[-1,1,0],[0,0,1],[1,0,0]]).inverse()]
        return CFAlgo2(P,M)
    
    
    # NEW ALGO N0.2 (left -> full, right -> full+rotation clockwise)
    @staticmethod
    def newalgo2():
        P = [[(1, 0, 0), (1, 1, 0), (0, 0, 1)],[(0, 1, 0), (1, 1, 0), (0, 0, 1)]]
        M = [matrix(QQ,[[1,-1,0],[0,1,0],[0,0,1]]).inverse(),matrix(QQ,[[-1,1,0],[0,0,1],[1,0,0]]).inverse()]
        return CFAlgo2(P,M)
    
    # NEW ALGO NO.3 (non-primitive!)
    @staticmethod
    def newalgo3():
        P = [[(1, 0, 0), (1, 1, 0), (1, 0, 1)],[(0, 0, 1), (1, 1, 0), (1, 0, 1)],[(0, 0, 1), (1, 1, 0), (0, 1, 1)],[(0, 1, 0), (1, 1, 0), (0, 1, 1)]]
        M = [matrix(QQ,[[1,-1,-1],[0,1,0],[0,0,1]]).inverse(),matrix(QQ,[[1,0,0],[0,1,0],[-1,1,1]]).inverse(),matrix(QQ,[[1,0,0],[0,1,0],[1,-1,1]]).inverse(),matrix(QQ,[[1,0,0],[-1,1,-1],[0,0,1]]).inverse()]
        return CFAlgo2(P,M)
    
    
    # NEW ALGO NO.4
    @staticmethod
    def newalgo4():
        P = [[(1, 0, 0), (1, 1, 0), (1, 1, 1)],[(0, 0, 1), (1, 1, 1), (1, 0, 0)],[(0, 0, 1), (1, 1, 0), (0, 1, 0)]]
        M = [matrix(QQ,[[1,0,-1],[0,1,-1],[0,0,1]]).inverse(),matrix(QQ,[[0,1,0],[1,0,0],[0,-1,1]]).inverse(),matrix(QQ,[[1,0,0],[-1,1,0],[0,0,1]]).inverse()]
        return CFAlgo2(P,M)
    
    # NEW ALGO NO.5
    @staticmethod
    def newalgo5():
        P = [[(1, 0, 0), (1, 1, 0), (1, 1, 1)],[(0, 0, 1), (1, 1, 1), (1, 0, 0)],[(0, 0, 1), (1, 1, 1), (0, 1, 0)],[(0, 1, 0), (1, 1, 1), (1, 1, 0)]]
        M = [matrix(ZZ,[[1,-1,1],[0,1,0],[0,0,1]]).inverse(),matrix(ZZ,[[1,0,0],[0,1,0],[0,-1,1]]).inverse(),matrix(ZZ,[[1,0,0],[0,1,0],[-1,0,1]]).inverse(),matrix(ZZ,[[1,0,0],[-1,1,1],[0,0,1]]).inverse()]
        return CFAlgo2(P,M)
    

cf_algos = Algorithms()


  

from slabbe.matrices import M3to2,M4to2,M4to3,perron_right_eigenvector
from slabbe import TikzPicture


class CFAlgo2(object):
    
    def __init__(self,P,M,sort=False):
        
        self._sim = []
        self._mat = []
        self._matinv = []
        self._images = []
        self._imagesvec = []
        self._sort = sort  
                
        m = len(P)
        self._simnum = m
        self._dim = len(P[0][0])
            
        for i in range(m):
            self._sim.append(Polyhedron(rays=[v for v in P[i]]))
            self._mat.append(M[i])
            self._matinv.append(M[i].inverse())
            R = [M[i].inverse()*vector(v) for v in P[i]]
            self._images.append(Polyhedron(rays=R))
            self._imagesvec.append(R)
        
        self._conevectors = [[x.vector() for x in self._sim[i].rays()] for i in range(m)]
        self._simplexvectors = [[x/sum(x) for x in y] for y in self._conevectors]
        self._simimvec = [[x/sum(x) for x in y] for y in self._imagesvec]
        
        
        S = set()
        for D in self._simplexvectors:
            for v in D:
                v.set_immutable()
                S.add(v)
        for D in self._simimvec:
            for v in D:
                v.set_immutable()
               
        for i in range(m):
            for j in range(i):
                if self._sim[i].intersection(self._sim[j]).dimension() == 3:
                    raise ValueError("Simplex %d and %d intersect" % (i,j))
            if set(self._simimvec[i]).issubset(S) == 0:
                raise ValueError("Simplex %d is mapped out of the others" % i)
        #TODO: is Markov        

    def plot_normsim(self,rays,col):
        proj = M3to2
        if col == 0:
            return polygon([proj*vector(t)/sum(vector(t)) for t in rays],fill=false,axes=false) 
        else:
            return polygon([proj*vector(t)/sum(vector(t)) for t in rays],color=col,fill=true,axes=false) 
            

    def plot_simplices(self):
        """
        Plot the simplices where the algorithm is defined on
        
        EXAMPLE::
        
            sage: B = cf_algos.brun()
            sage: B.plot_simplices()
        """
        G = Graphics()
        proj = M3to2
        col = rainbow(len(self._sim),'rgbtuple')
        for i,D in enumerate(self._sim):
            G += polygon([proj*t.vector()/sum(t.vector()) for t in D.rays()],fill=false,axes=false,rgbcolor=col[i],legend_label=str(i)) 
        return G
    
    
    def plot_union_simplices(self,f=false):
        
        proj = M3to2
        S = sum(self._sim)
        return polygon([proj*t.vector()/sum(t.vector()) for t in S.rays()],fill=f,axes=false) 

#proj = M4to3
#for i in range(len(P)):
#G += Polyhedron([proj()*vector(t) for t in P[i]]).plot(color=rainbow(len(P))[i],fill=false) 
 
    def localize_sim(self,M,L):
        return self.plot_normsim([M*vector(v) for v in L],col='red')

    def IFS(self,list_matrices,Lsim,n_iter):
        G = Graphics()
        for k in range(n_iter):
            Lfin = []
            for M in list_matrices:
                for S in Lsim:
                    Lfin.append([M*vector(v) for v in S])
            Lsim = Lfin
        for j in range(len(Lsim)):
            G += self.plot_normsim(Lsim[j],col='red')
        return G    

    def loc_line(self,M,L):
        return line([M3to2*M*vector(v)/sum(M*vector(v)) for v in L],color='black',thickness=5)
    
    
    def which_cone(self,v):
        """
        Return the label of the 3-dimensional cone in which the vector v belongs to
        or -1 if not in the domain
        
        EXAMPLE::
        
            sage: B = cf_algos.newalgo1()
            sage: v = (-1,.7,0)
            sage: B.which_cone(v)
            -1
            
            sage: v = (56,32,117)
            sage: B.which_cone(v)
            0
        """    
        for i in range(len(self._sim)):
            if v in self._sim[i]:
                return i
        return -1
    
     
        
    def proj_orbit(self,vec,n_iter,connect_lines=false):

        if self._sort == 1:
            vl = list(vec)
            vl.sort(reverse=true)
            w = vl
            x = vector(w)/sum(w)
        else:
            x = vector(vec)/sum(vec)
        dim = len(x)
        O = [x]
        O1 = []
        W = []
        
        for n in range(n_iter):
            j = self.which_cone(x)
            if j != -1:
                fx = self._matinv[j]*x
                y = fx/sum(fx)
                O1.append([x,y])
                O.append(y)
                W.append(j)
                x = y
            else:
                break
        W.append(self.which_cone(x))
        if connect_lines:
            return O,O1,W 
        else:
            return O,W

    
    def cone_orbit(self,vec,n_iter):

        if self._sort == 1:
            vl = list(vec)
            vl.sort(reverse=true)
            x = vector(vl)
        else:
            x = vector(vec)
        dim = len(x)
        O = [x]
        W = []
        for n in range(n_iter):
            j = self.which_cone(x)
            if j != -1:
                fx = self._matinv[j]*x
                O.append(fx)
                W.append(j)
                x = fx
            else:
                break
        W.append(self.which_cone(x))
        return O,W 
    
    def proj_orbit_plot(self,vec,n_iter,show_sim=1,connect_lines=false):
        
        G = Graphics()
        proj = M3to2
        if connect_lines:
            O,O1,W = self.proj_orbit(vec,n_iter,connect_lines)
            L = [(proj*O1[i][0],proj*O1[i][1]) for i in range(len(O1))]
            P = [proj*O[i] for i in range(len(O))]
            for x in L:
                G += line(x,color='black')
            if show_sim == 1:
                return self.plot_simplices() + G + point(P)
            else:
                return self.plot_union_simplices() + G + point(P)
        else:
            O,W = self.proj_orbit(vec,n_iter,connect_lines)
            P = [proj*O[i] for i in range(len(O))]
            if show_sim == 1:
                return self.plot_simplices() + point(P)
            else:
                return self.plot_union_simplices() + point(P)
       
           
    def cf_dict(self):

        D1 = {}
        proj = M3to2
        
        for k in range(self._simnum):
            D1[k] = []
            for i in range(self._simnum):
                if all([x in self._images[k] for x in self._sim[i].rays()]):
                    D1[k] += [i]
        return D1
                
        
    def cf_automaton(self,init='all'):
        
        L = []
        D1 = self.cf_dict()
        for i in D1.keys():
            for j in range(len(D1[i])):
                L += [(str(i),str(D1[i][j]),i,i)]
        if init == 'all':
            return Automaton(L,initial_states=[str(i) for i in D1.keys()],final_states=[str(i) for i in D1.keys()])
        else:
            return Automaton(L,initial_states=[str(init)],final_states=[str(i) for i in D1.keys()])

    
    def automaton_lang_mat_it(self,n,length_n=1,state='all'):
        
        a = self.cf_automaton(init=state)
        a1 = a.transposition()
        for x in a1.language(n):
            prod = 1
            if length_n:
                if len(x) == n:
                    for j in range(1,len(x)):
                        prod = prod*self._mat[x[-j]]
                    yield x,prod
            else:
                if len(x) > 0:
                    for j in range(1,len(x)):
                        prod = prod*self._mat[x[-j]]
                    yield x,prod
                #else:
                #    yield x,1
                    
                
                    
    def cf_cylinders(self,n,length_n=1,state='all'):
        
        G = Graphics()
        
        lang,mat = zip(*self.automaton_lang_mat_it(n,length_n,state))
        for i in range(len(lang)):
            R = [mat[i]*vector(r) for r in (self._sim[lang[i][0]]).rays_list()]
            G += self.plot_normsim(R,0)
        return G    
    
    
    def cf_fractal(self,sim_rays,n):
        
        G = Graphics()
        G += self.plot_normsim(sim_rays,col='white')
        
        lang,mat = zip(*self.automaton_lang_mat_it(n,0,'all'))
       
        Madm = [mat[i] for (i,x) in enumerate(lang)]
        
        for M in Madm:
            if M != 1:
                M.set_immutable()
        Madm1 = list(set(Madm))
        for M in Madm1:
        
            R = [M*vector(r) for r in sim_rays]
            G += self.plot_normsim(R,col='white')
        
        return self.plot_union_simplices(f=true) + G
        
    
        
    def cf_eigenvectors(self,n,length_n=1,state='all'):
    
        Lfin = []
        lang,mat = zip(*self.automaton_lang_mat_it(n,length_n,state))
        MLoops = [mat[i] for (i,x) in enumerate(lang) if x[0] == x[-1] if len(x)>=2]
        for M in MLoops:
            ev = perron_right_eigenvector(M)[1] 
            nev = ev/sum(ev)
            nev.set_immutable()
            Lfin += [nev]
        return list(set(Lfin))  
        # to check why I have multiple identical eigenvectors (for this reason i make a list(set()) )


    def cf_eigenvectors_plot(self,n,length_n=1,state='all'):
    
        proj = M3to2
        return point([proj*vector(t) for t in self.cf_eigenvectors(n,length_n,state)],size=1,axes=false)
   
    
    
                      
                    
                   
               