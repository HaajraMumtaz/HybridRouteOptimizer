from genetic_algo.src.main import run_full_ga
from maps_api.main import get_top_best


# main.py
def main():
    top_3=run_full_ga()

    get_top_best(top_3)


if __name__ == "__main__":
    main()
