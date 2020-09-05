import datetime

from dask_benchmarks import run_benchmark, setup

NBYTES = [10e5, 10e6, 10e7, 10e8, 10e9, 10e10]
NPARTITIONS = [50, 100, 250, 500, 1000, 2500]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--sort", action="store_true")
    args = parser.parse_args()

    operation = "sort" if args.sort else "groupby"

    now = datetime.datetime.now().isoformat()
    output_filename = f"outputs/shuffle_run_{now}.csv"

    setup(is_dask=True, is_ray=True)

    print(f"Running benchmarks for {operation}.")
    for nbytes in NBYTES:
        for npartitions in NPARTITIONS:
            print(
                f"\n\nRunning {operation} benchmarks for nbytes={nbytes:.2e},"
                f" npartitions={npartitions:.2e}")
            print(f"\nRunning Ray benchmark.\n")
            run_benchmark(
                int(nbytes),
                npartitions,
                num_nodes=args.num_nodes,
                sort=args.sort,
                is_ray=True,
                output_filename=output_filename,
            )
            print(f"\nRunning Dask benchmark.\n")
            run_benchmark(
                int(nbytes),
                npartitions,
                num_nodes=args.num_nodes,
                sort=args.sort,
                is_dask=True,
                output_filename=output_filename,
            )
