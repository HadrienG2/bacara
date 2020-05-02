# De l'art de scanner `usage_bitmap`

En cours d'allocation, on veut parcourir usage_bitmap pour chercher une zone mémoire non utilisée. Pour garder le temps d'allocation borné, on ne veut le parcourir qu'une seule fois, et déclarer forfait si on n'y trouve rien.

Mais dans quel ordre doit-on parcourir les superblocs du bitmap ?

## Scan simple

• Parcours des indices de 0 à N
• Algorithme actuel
• Algorithme le plus simple
• Compacité optimale, fragmentation minimale
• Problème: Pression inhomogène sur les indices: indices vers 0 sur-utilisés (donc beaucoup de contention), indices vers N sous-utilisés mais considérés par le scan en dernier.

## Scan aléatoire

• On tire un indice R au pif dans 0..N, puis on scanne R..N suivi de 0..R+S-1 avec S le nombre de superblocs requis (légèrement redondant, mais évite de traiter le premier trou de R..N de façon spéciale)
• Attention à nettoyer les variables d'état quand on repart de 0.
• Distribution optimale des accès multithread sur le bitmap.
• Petit coût du RNG, probablement négligeable devant les gains autres.
• Problème: Accès éparpillés = fragmentation des allocations = difficile d'allouer des gros blocs.

## Scan guidé

• On introduit une variable partagée X, init. 0
• Les threads scannent X..N suivi de 0..X+S-1, comme précédemment donc mais en suivant X partagé au lieu de R aléatoire.
• Le thread qui alloue change X pour indiquer aux autres threads où chercher leur propre allocation. Je peux imaginer trois variantes...
    1. Le thread maintient les autres threads à distance au fil de sa recherche. A chque indice I qu'il considère, il place X en I+S.
        • Garantit une interférence minimale entre threads concurrents.
        • Mais demande un très grand nombre d'écritures sur X partagée pour ça, donc probablement pas rentable.
        • Surtout que les collisions d'allocation ne sont en principe pas si courantes que ça...
    2. Lorsque le thread trouve un trou I..I+S, il modifie X en I+S avant d'entamer l'allocation.
        • La protection contre les interférences n'est pas assurée pendant la recherche, seulement pendant l'allocation.
        • Mais la recherche est plus rapide, ce qui limite aussi le risque d'interférence.
        • Au lieu de O(nb cases lues) écritures sur X, on en fait O(nb trous considérés), c'est déjà beaucoup mieux
    3. Après une allocation I..I+S réussie, le thread modifie X en ISS
        • Aucune protection contre les interférences
        • Au plus 1 écriture sur X garantie
• Dans tous les cas:
    • Usage "ring buffer" du stockage de l'allocateur, plus homogène
    • Propriétés de compacité moins bonne si les allocations sont inhomogènes, mais pas aussi horribles que aléatoire

## Conclusion

Il me semble clair qu'un benchmark sera necessaire pour comparer ces stratégies.
