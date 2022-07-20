from pickle import TRUE
import string
from numpy import true_divide
from search import *
import random
from random import *
from datetime import datetime
from time import sleep
import cv2 as cv
import matplotlib.pyplot as plt

#variabili globali finestra disegna_mappa
windows_name='mappa'
windows=None
grandezza_quadrati=(55,55)
dimX_finestra=None
dimY_finestra=None

#variabili globali di sistema
dimX=0
dimY=0
colori=['Y','G','B']


def costo(colore):
        if colore=='B':
            return 1
        elif colore=='Y':
            return 2
        elif colore=='G':
            return 3
        elif colore=='0':
            return 0

def crea_goal(initial):
    i=0
    goal=list()
    goal1=list()
    goal2=list()
    goal3=list()
    for i in initial:
        if i[1]!='T':
            goal1.append(('B',False))
        else:
            goal1.append(('0','T'))
    for i in initial:
        if i[1]!='T':
            goal2.append(('G',False))
        else:
            goal2.append(('0','T'))
    for i in initial:
        if i[1]!='T':
            goal3.append(('Y',False))
        else:
            goal3.append(('0','T'))
    goal.append(goal1);goal.append(goal2);goal.append(goal3)
    return goal

class UniformColoring(Problem):
    def __init__(self, initial, m, n, goal=None):
        """Specifica lo stato iniziale, i possibili stati goal
        e il numero di righe e di colonne della matrice di input di grandezza mxn"""
        super().__init__(initial, goal)
        self.m=m
        self.n=n

    def get_index_t(self,state):
        """
        state è una lista
        Ritorna l'indice di T nello stato state """
        count=0
        for (elemento,testina) in state:
            if(testina=='T'):
                return count
            count+=1
        return False

    def get_origine_t(self):
        """
        Ritorna l'indice della cella d'origine di T
        """
        state=self.goal[1]
        count=0
        for (elemento,testina) in state:
            if(testina=='T'):
                return count
            count+=1
        return count

    def actions(self, state):
        """
        Restituisce tutte le azioni che possono essere eseguite nello stato specificato (UP,DOWN,LEFT,RIGHT)
        e tutte le colorazioni possibili associate ad ogni movimento

        olte ciò viene restituito anche se le colorazioni devono essere effettuate oppure no
        le restituisce sotto forma di tupla

        una possibile forma può essere:
        [
            ('U', ('Y', False)),
            ('U', ('G', True)),
            ('U', ('B', True)),
            ('R', ('Y', True)),
            ('R', ('G', False)),
            ('R', ('B', True))
        ]
        (1,(2,3)):
        1 --> direzione
        2 --> colorazione possibile
        3 --> se true allora la cella non ha lo stesso colore della colorazione
            e non è la cella iniziale della testina
            se false --> allora non va colorata la cella poiche puo essere la cella iniziale
        """
        lista_stato=state.state
        colori=['Y','G','B']
        lista_azioni=[]
        delta = {'U': -self.n, 'D': +self.n, 'L': -1, 'R': +1}
        possible_action_move=['U','D','L','R']
        #print("state",state)

        indice_testina=self.get_index_t(lista_stato)

        #azioni possibili
        if indice_testina % self.n ==0:
            possible_action_move.remove('L')
        if indice_testina < self.n:
            possible_action_move.remove('U')
        if indice_testina % self.n == self.n-1:
            possible_action_move.remove('R')
        if indice_testina >= (self.n * (self.m-1)):
            possible_action_move.remove('D')
        if state.action != None :
            azione_precedente=state.action[0]
            if(azione_precedente=='U'):
                possible_action_move.remove('D')
            if(azione_precedente=='L'):
                possible_action_move.remove('R')
            if(azione_precedente=='R'):
                possible_action_move.remove('L')
            if(azione_precedente=='D'):
                possible_action_move.remove('U')

        #se il colore della cella è !=0 allora devo colorare dello stesso colore
        if(lista_stato[indice_testina][0]=='0' and lista_stato==self.initial):
            #se è uguale a 0 allora è la prima cella o il goal test
            #nel caso sia il goal test allora ci pensera l'algoritmo
            #se è la prima cella allora posso scegliere qualunque colore
            for value in possible_action_move:
                stato_vicino=lista_stato[(indice_testina+delta[value])]
                for col in colori:
                    if(col== stato_vicino[0] ):
                        """ se uguale a False allora non c'è bisogno di colorarlo"""
                        lista_azioni.append( (value,(col,False)) )
                    else:
                        lista_azioni.append( (value,(col,True)) )
        else:
            colorazione=lista_stato[indice_testina][0]

            for value in possible_action_move:
                stato_vicino=lista_stato[(indice_testina+delta[value])]
                if(colorazione== stato_vicino[0] ):
                    """ se uguale a False allora non c'è bisogno di colorarlo"""
                    lista_azioni.append( (value,(colorazione,False)) )
                else:
                    lista_azioni.append( (value,(colorazione,True)) )
        return lista_azioni

    def result(self, state, action):
        """Restituisce lo stato che risulta dall'esecuzione dell'azione data nello stato indicato.
        L'azione deve essere una di self.actions(state).
        """
        testina=self.get_index_t(state)
        colorazione=action[1]
        new_state = list(state)
        delta = {'U': -self.n, 'D': +self.n, 'L': -1, 'R': +1}
        neighbor = testina + delta[action[0]]

        """colorazione e spostamento"""

        if(new_state[neighbor][0]=='0'):
            new_state[testina],new_state[neighbor]=(new_state[testina][0],False),("0", "T")
        else:
            new_state[testina],new_state[neighbor]=(new_state[testina][0],False),(colorazione[0], "T")
        return tuple(new_state)

    def goal_test(self, state):
        " Ritorna True se tutte le celle sono dello stesso colore, False altrimenti"
        return self.get_index_t(state)==self.get_origine_t() and self.tutti_colorati(state)

    def h(self,state):
        nodo = state.state
        n_elementi=len(nodo)
        contatore={'B':0,'Y':0,'G':0}
        costi=[]
        for elemento in nodo:
            if(elemento[1] != 'T'):
                if(elemento[0]!='0'):
                    contatore[elemento[0]]+=1
        for colore in contatore.keys():
            a=n_elementi - contatore[colore]
            costi.append(a*costo(colore) + a)
        return min(costi)

    def path_cost(self, c, state1, action, state2):
        """Restituisce il costo di un percorso di soluzione che arriva allo stato2 da stato1 tramite action,
        assumendo il costo c per arrivare allo stato1.
        Se il problema è tale che il percorso non ha importanza, questa funzione esaminerà solo lo stato2.
        Se il percorso è importante, prenderà in considerazione c e forse state1 e action.
        Il metodo predefinito costa 1 per ogni passaggio nel percorso."""
        colorazione=action[1]
        if colorazione[1] == True:
            return c+costo(colorazione[0])+1
        else:
            return c+1


    def tutti_colorati(self,state):
        celle_blu=0
        celle_verdi=0
        celle_gialle=0
        for cella in state:
            if cella[1]!='T':
                if cella[0]=='B':
                    celle_blu+=1
                elif cella[0]=='G':
                    celle_verdi+=1
                elif cella[0]=='Y':
                    celle_gialle+=1
        # se tutti le celle sono dello stesso colore -1 perchè tengo conto di T
        if celle_blu == dimX*dimY-1:
            return True
        elif celle_gialle == dimX*dimY-1:
            return True
        elif celle_verdi == dimX*dimY-1:
            return True
        else:
            return False

""" ----------------    funzioni di disegno     -----------------"""

def disegna_mappa_matrice(matrice,windows=False,titolo_finestra="titolo"):
    """
    se windows è false allora non è stata passata la finestra su cui disegna_mappa_matrice
    allora la creo temporanea
    """
    if(windows!=False):
        return 'todo'
    for stati in matrice:
        disegna_mappa(stati,titolo_finestra,True)

def disegna_mappa(matrice,nome_stato,crea_finestra=True,w=None,distanza_bordo=(50,50)):
    """
    w è la finestra
    disegna in una finestra la matrice data in input.
    Quando crea_finestra è settatto uguale a True allora
    inizializza la finestra alle impostazioni di base e fa ritornare il puntatore alla finestra.
    la distanza_bordo(x,y) rappresenta l'origine di dove sara creato il primo quadrato,
    di DEFAULT è settato a (10,10)
    """
    posizione = (distanza_bordo[0],distanza_bordo[1])
    verde=(0,255,0)
    blu=(0,0,255)
    rosso=(255,0,0)
    giallo=(255,255,0)
    if(crea_finestra==True):
        #inizializza la finestra in cui disegnare
        dimX_finestra=dimX*grandezza_quadrati[0]+100
        dimY_finestra=dimY*grandezza_quadrati[1]+100
        w=np.zeros((dimX_finestra,dimY_finestra,3),dtype='uint8')
        #print(w)
        #plt.imshow(w)##il primo è il nome della finestra, il secondo una matrice che rappresenta la finestra
        #plt.show()
        #imposto il colore di default a bianco
        w[:]=255,255,255
        """verifico se matrice è una lista o una matrice,
            se è una lista cambio il metodo di eseguzione del for

        """
    cv.putText(w,nome_stato,(50,30),cv.FONT_HERSHEY_DUPLEX,0.6,(0,0,0),1)
    if(type(matrice).__name__=='Node' ):


        matrice=matrice.ritorna_lista()
    elif(type(matrice).__name__=="tuple"):
        matrice=list(matrice)

    elif(type(matrice).__name__=='list'):
        #allora è una lista

        colonna=0
        for elemento,testina in matrice:
            if(elemento=="G"):##verifico il colore e lo disegno
                crea_quadrato_con_cornice(w,grandezza_quadrati,posizione,verde)
            elif(elemento=="B"):
                crea_quadrato_con_cornice(w,grandezza_quadrati,posizione,blu)
            elif(elemento=="Y"):
                crea_quadrato_con_cornice(w,grandezza_quadrati,posizione,giallo)
            elif(elemento == '0'):##siamo nella cella T
                crea_quadrato_con_cornice(w,grandezza_quadrati,posizione,rosso)


            if(testina!=False):
                #disegna testina
                cv.putText(w,"T",(posizione[0]+20,posizione[1]+30),cv.FONT_HERSHEY_DUPLEX,0.6,(0,0,0),1)


            posizione=(posizione[0]+grandezza_quadrati[0],posizione[1])
            colonna+=1
            if(colonna==dimY):
                colonna=0
                posizione=(posizione[0]-grandezza_quadrati[0]*dimY,posizione[1]+grandezza_quadrati[1])

    else:
        for righe in matrice:
            for colonne in righe:
                if(colonne=="G"):##verifico il colore e lo disegno
                    crea_quadrato_con_cornice(w,grandezza_quadrati,posizione,verde)
                elif(colonne=="B"):
                    crea_quadrato_con_cornice(w,grandezza_quadrati,posizione,blu)
                elif(colonne=="Y"):
                    crea_quadrato_con_cornice(w,grandezza_quadrati,posizione,giallo)
                else:##siamo nella nell'origine
                    crea_quadrato_con_cornice(w,grandezza_quadrati,posizione,rosso)

                """
                se l'origine si trova (10,10),
            i   l quadrato successivo si troverà alle cordinate
                       (10 + grandezza_quadrato[0],10)
                """
                posizione=(posizione[0]+grandezza_quadrati[0],posizione[1])

            """se abbiamo finito le colonne ci troveremo alle cordinate
                                        (10+grandezza_quadrati[0]*len(righe))
                devo andare alla coordinata (10, 10+grandezza_quadrati[1])
                quindi tolgo dalla coordinata x la grandezza_quadrati[0]*numero_di_quadrati_generati
            """
            posizione=(posizione[0]-grandezza_quadrati[0]*(len(righe)),posizione[1]+grandezza_quadrati[1])

    #mostro la mappa
    plt.yticks([])
    plt.xticks([])
    plt.imshow(w)##il primo è il nome della finestra, il secondo una matrice che rappresenta la finestra
    plt.show()
    if(crea_finestra==True):
        return w

def crea_quadrato_con_cornice(windows,grandezza_quadrato,posizione,colore,colore_bordo=(0,0,0)):
    windows[posizione[1]:posizione[1]+grandezza_quadrato[0],posizione[0]:posizione[0]+grandezza_quadrato[1]]=colore
    cv.rectangle(windows,(posizione[0],posizione[1]),(posizione[0]+grandezza_quadrato[1],posizione[1]+grandezza_quadrato[0]),colore_bordo,thickness=2)


""" ----------------    algoritmi non informati     -----------------"""

def depth_first_graph_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(tuple(node.state))
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None

def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    count=0
    while frontier:
        node = frontier.popleft()
        count=+1
        explored.add(tuple(node.state))
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(list(child.state)):
                    return child
                frontier.append(child)
    return None

def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)

    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(list(node.state)):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        """
        aggiunge il nodo alla lista degli esplorati
        e va a cercare nella frontiera
        """
        explored.add(tuple(node.state))
        count=0
        for child in node.expand(problem):
            count+=1
            """
            se non è negli esplorati e non è nella frontiera allora lo aggiungo alla frontiera
            vuol dire che non l'avevo trovato prima
            se invece si trova nella frontiera vado ad richiama
            """
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None

def uniform_cost_search(problem, display=TRUE):
    return best_first_graph_search(problem, lambda node: node.path_cost, display)

def depth_limited_search(problem, limit=10):

    def recursive_dls(node, problem, limit):
        if problem.goal_test(list(node.state)):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)

def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):

        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result

""" ----------------    algoritmi euristici     -----------------"""

def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)
