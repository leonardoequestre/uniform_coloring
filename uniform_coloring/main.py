from trova_matrice  import *
from uniform_coloring import *
import uniform_coloring as uniC
dimensioniMatrice=None
lista_numeri=None


def main():
    global dimensioniMatrice
    global lista_numeri
    lista_colori=[]


    print("\nrichiesta input utente:\n")
    immagine,dimensioniMatrice=preleva_input()
    plt.imshow(immagine)
    plt.show()

    print("\nritaglio l'immagine\n")
    immagine=ritaglia_immagine(immagine)
    plt.imshow(immagine)
    plt.show()

    print("\ncerco le lettere\n")
    immagine, lettere_trovate=trova_lettere(immagine)
    if(type(immagine).__name__!="ndarray"):
        print("l'immagine non ha letto tutte le lettere quindi esco... \nQUIT\n\n")
        exit()

    print("\navvio rete neurale per capire le lettere trovate\n")
    lista_numeri=lettura_rete_neurale()

    for elemento in lista_numeri:
        if elemento==0:
            lista_colori.append('B')
        elif elemento==1:
            lista_colori.append('G')
        elif elemento==2:
            lista_colori.append('T')
        elif elemento==3:
            lista_colori.append('Y')

    mat_input=np.zeros((dimensioniMatrice[0],dimensioniMatrice[1]))
    mat_input=np.array(mat_input,dtype=str)
    contatore_lista=0
    x=0
    y=0
    while x<dimensioniMatrice[0]:
        y=0
        while y<dimensioniMatrice[1]:
            mat_input[x,y]=lista_colori[contatore_lista]
            contatore_lista+=1
            y+=1
        x+=1

    print("mat_input: ",mat_input)

    # cancello i file dalla cartella output
    print("\ncancello i file in output\n")
    img_number=0
    while os.path.isfile(f"./immagini/output/immagine{img_number}.jpeg"):
        file_path = Path(f'./immagini/output/immagine{img_number}.jpeg')
        try:
            file_path.unlink()
        except OSError as e:
            print("Error: %s : %s" % (file_path, e.strerror))
        img_number+=1
    print("file in output cancellati\n")

    # inizio uniform coloring


    print("\n\navvio uniformColoring\n\n")
    initial=list()
    # creo lo stato iniziale
    for x in mat_input:
        for y in x:
            if y== 'T':
                initial.append(('0','T'))
            else:
                initial.append((y,False))

    #inizializzo le dimensioni di uniform_coloring
    uniC.dimX=dimensioniMatrice[0]
    uniC.dimY=dimensioniMatrice[1]

    # inizializzo windows


    goal=crea_goal(initial)
    problemUC=UniformColoring(initial,dimensioniMatrice[0],dimensioniMatrice[1],goal)


    print(f"si parte...................\n stato della lista iniziale: \n{initial}")
    print(f"\n dimensione matrice: {dimensioniMatrice[0]} x {dimensioniMatrice[1]}\n\n")
    disegna_mappa(initial,"stato iniziale",True)

    while True:
        print("inserisci algoritmo da usare:\n\
        A: Astar search algoritm euristico\n\
        BFS: Breadth first search\n\
        DFS: Depth first search\n\
        DLS: Deapth limited search\n\
        IDS: Iterative deepening search\n\
        UCS: Unifrom cost search\n\
        Q: quit\n")
        algoritmo=input()
        algoritmo=algoritmo.upper()
        if(algoritmo=='A'):
            print("partenza A*")
            tempo1=datetime.now()
            ricerca_astar=astar_search(problemUC)
            tempo2= datetime.now()
            print("\n A* search, azioni eseguite",ricerca_astar.solution())
            print("\n costo= ",ricerca_astar.path_cost)
            print("\ntempo A*= ",tempo2-tempo1)
            disegna_mappa(initial,"INITIAL",True)
            disegna_mappa_matrice(ricerca_astar.solution(),titolo_finestra="A*")
        elif(algoritmo=='BFS'):
            print("partenza BFS")
            tempo1 = datetime.now()
            ricerca_bfs=breadth_first_graph_search(problemUC)
            tempo2 = datetime.now()
            print("\n breadth_first_search ",ricerca_bfs.solution())
            print("\n costo= ",ricerca_bfs.path_cost)
            print("tempo bfs= ",tempo2-tempo1)
            disegna_mappa(initial,"INITIAL",True)
            disegna_mappa_matrice(ricerca_bfs.solution(),titolo_finestra="BFS")

        elif(algoritmo=='DFS'):
            print("partenza DFS")
            tempo1 = datetime.now()
            ricerca_profondita=depth_first_graph_search(problemUC)
            tempo2 = datetime.now()
            print("\ndepth_first_search ",ricerca_profondita.solution())
            print("\n costo= ",ricerca_profondita.path_cost)
            print("tempo dfs= ",tempo2-tempo1)
            disegna_mappa(initial,"INITIAL",True)
            disegna_mappa_matrice(ricerca_profondita.solution(),titolo_finestra="DFS")
        elif(algoritmo=='DLS'):
            print("partenza DLS")
            l=(dimensioniMatrice[0]+dimensioniMatrice[1])
            tempo1 = datetime.now()
            ricerca_profondita_limitata=depth_limited_search(problemUC,l+l)
            tempo2 = datetime.now()
            if(type(ricerca_profondita_limitata).__name__=="str"):
                print(f"superata la profondità massima:\n\t {ricerca_profondita_limitata} limit {l}")
                print("tempo dls= ",tempo2-tempo1)
            else:
                print("\ndepth_limited_search ",ricerca_profondita_limitata.solution())
                print("\n costo= ",ricerca_profondita_limitata.path_cost)
                print("tempo dls= ",tempo2-tempo1)
                disegna_mappa(initial,"INITIAL",True)
                disegna_mappa_matrice(ricerca_profondita_limitata.solution(),titolo_finestra="DLS")
        elif(algoritmo=='IDS'):
            print("partenza ITERATIVE DEEPENING SEARCH")
            tempo1=datetime.now()
            ricerca_ids=iterative_deepening_search(problemUC)
            tempo2= datetime.now()
            print("\n IDS azioni eseguite: ",ricerca_ids.solution())
            print("\n costo= ",ricerca_ids.path_cost)
            print("\n tempo ids= ",tempo2-tempo1)
            disegna_mappa(initial,"INITIAL",True)
            disegna_mappa_matrice(ricerca_ids.solution(),titolo_finestra="IDS")
        elif(algoritmo=='UCS'):
            print("partenza UNIFORM COST SEARCH")
            tempo1=datetime.now()
            ricerca_uniform=uniform_cost_search(problemUC)
            tempo2 = datetime.now()
            print("\n UCS: ",ricerca_uniform.solution())
            print("\n costo= ",ricerca_uniform.path_cost)
            print("\n tempo ucs= ",tempo2-tempo1)
            disegna_mappa(initial,"INITIAL",True)
            disegna_mappa_matrice(ricerca_uniform.solution(),titolo_finestra="UCS")
        elif(algoritmo=='Q'):
            print("\n\t\t\tUSCITA DAL PROGRAMMA IN CORSO\n\n\n")
            break;
        else:
            print(f"\n\nERRORE: il valore inserito non è valido reinserire\n\n")































if __name__=="__main__":
    main()
