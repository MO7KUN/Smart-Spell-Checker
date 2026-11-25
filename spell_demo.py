"""Small French noisy-channel demonstration. Run with Python 3.9 or later."""
import argparse
from collections import Counter
from math import log1p
from unicodedata import normalize

# Hand-authored teaching examples (observed spelling, intended spelling).
TRAIN = """
eleve élève
ecole école
etude étude
etudiant étudiant
annee année
idee idée
mere mère
pere père
apres après
succes succès
oublie oublié
marche marché
foret forêt
hopital hôpital
garcon garçon
lecon leçon
facade façade
""".splitlines()[1:]

TEST = """
bonjor bonjour
bibliotheque bibliothèque
universite université
bibliohtèque bibliothèque
faculte faculté
carriere carrière
etage étage
chateau château
francais français
recette recette
plage plage
cour cours
marie marié
porte porté
livr livre
stgae stage
""".splitlines()[1:]

# Closed candidate vocabulary shared by both decoders.
WORDS = sorted(set("""
élève école étude étudiant année idée mère père après succès oublié marché forêt
hôpital garçon leçon façade bonjour bibliothèque université faculté carrière étage
château français recette plage cour cours marie marié porte porté livre stage
salut journée soir univers bureau conseil papier livret courbe recette page
""".split()))


def pairs(lines):
    return [tuple(line.split()) for line in lines]


def learned_costs():
    substitutions = Counter()
    for observed, intended in pairs(TRAIN):
        if len(observed) != len(intended):
            raise ValueError("Training substitutions require aligned examples")
        substitutions.update((a, b) for a, b in zip(observed, intended) if a != b)
    return {pair: 1 / (1 + log1p(count)) for pair, count in substitutions.items()}


def distance(source, target, costs):
    previous = list(range(len(target) + 1))
    for row, a in enumerate(source, 1):
        current = [row]
        for col, b in enumerate(target, 1):
            substitution = 0 if a == b else costs.get((a, b), 1)
            current.append(min(previous[col] + 1, current[-1] + 1,
                               previous[col - 1] + substitution))
        previous = current
    return previous[-1]


def predict(word, costs):
    word = normalize("NFC", word.strip().lower())
    if not word or any(ch.isspace() for ch in word):
        raise ValueError("Enter one non-empty word")
    return min(WORDS, key=lambda candidate: (distance(word, candidate, costs), candidate))


def evaluate():
    test = pairs(TEST)
    assert set(target for _, target in test).isdisjoint(target for _, target in pairs(TRAIN))
    assert all(target in WORDS for _, target in test)
    for name, costs in [("Learned substitutions", learned_costs()), ("Levenshtein", {})]:
        results = [(word, target, predict(word, costs)) for word, target in test]
        correct = sum(target == guess for _, target, guess in results)
        print(f"{name}: {correct}/{len(test)} ({correct / len(test):.1%})")
        for word, target, guess in results:
            if guess != target:
                print(f"  {word} -> {guess} (expected {target})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("word", nargs="?")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        try:
            print(predict(args.word, learned_costs()))
        except ValueError as error:
            parser.error(str(error))
