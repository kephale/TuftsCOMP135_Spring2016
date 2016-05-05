
# Evolve a solution such that the array sums to the target

import random

target = 42

best_solution = [ 0, 0 ]

def candidate_score( c ):
    return target - sum( c )

step = 0
scores = []
while candidate_score(best_solution) > 0:
    mutant = [ el + round(random.random() * 2 - 1) for el in best_solution ]

    if candidate_score(mutant) < candidate_score(best_solution):
        best_solution = mutant

    print( "Step: %d, best solution is: %s, with score %d" % ( step, str(best_solution), candidate_score(best_solution) ) )

    scores += [ candidate_score(best_solution) ]
    step += 1
