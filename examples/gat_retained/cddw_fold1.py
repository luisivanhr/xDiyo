"""Explicit archived time_B/CD_DW fold_1 recipe; choose seed 17, 42 or 73."""
from .replay import prepare, main
FAMILY = 'time_B/CD_DW'
FOLD = 'fold_1'
WINDOW = 10


def recipe(research_root, *, seed):
    return prepare(research_root, family=FAMILY, fold=FOLD, seed=seed, window=WINDOW)


if __name__ == '__main__':
    main(recipe)
