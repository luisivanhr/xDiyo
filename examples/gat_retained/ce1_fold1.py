"""Explicit archived time_C/E1 fold_1 recipe; choose seed 17, 42 or 73."""
from .replay import prepare, main
FAMILY = 'time_C/E1'
FOLD = 'fold_1'
WINDOW = None


def recipe(research_root, *, seed):
    return prepare(research_root, family=FAMILY, fold=FOLD, seed=seed, window=WINDOW)


if __name__ == '__main__':
    main(recipe)
