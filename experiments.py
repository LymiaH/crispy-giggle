import subprocess
import traceback
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
from timeit import default_timer as timer

import time

import os

from common import eprint

PARAMS = ['-q']
COMMAND = ['python', '-u', 'comparer.py']
WORKING_DIRECTORY = Path("../crispy-giggle/")


def qprint(*args, **kwargs):
    if not QUIET:
        eprint(*args, **kwargs)


def run_comparer(graph_path: str = "./paths/twointersect.txt", mode: str = "graph", test_case: str = "tri_simp",
                 thickness: int = 15):
    """
    Runs the comparer program.
    :param graph_path: Path to the graph to run experiments on
    :param mode: way/graph: way uses a DFS to generate sequential waypoints, graph produces nodes and edges
    :param test_case: no_simp/los_simp/tri_simp: Test case to use: No simplification, Line of Sight simplification, Triangle Simplification
    :param thickness: Thickness of the road (default 15)
    :return: (TIME_ELAPSED, CORRECT, MISSING, EXTRA)
    """
    args = []
    for arg in COMMAND: args.append(arg)
    args.append('-i')
    args.append(graph_path)
    args.append('-m')
    args.append(mode)
    args.append('-tc')
    args.append(test_case)
    args.append('-t')
    args.append(str(thickness))
    for arg in PARAMS: args.append(arg)

    qprint("[EXPERIMENT] Running: " + ' '.join(args))
    start = timer()
    p = subprocess.Popen(
        args,
        cwd=str(WORKING_DIRECTORY),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        text=True,
        shell=True
    )
    stdout, stderr = p.communicate()  # type: str, str
    end = timer()
    if stderr and len(stderr) > 0:
        qprint("[EXPERIMENT] Errors: " + stderr)
    qprint("[EXPERIMENT] Result: " + stdout)
    elapsed = end - start
    qprint("[EXPERIMENT] Time Elapsed: %ss" % elapsed)
    return tuple([elapsed] + [int(num) for num in stdout.strip().split(',')])


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("-q", "--quiet", action='store_true', required=False,
                      help="Don't print information to stderr.")
    def int_gte_zero(param: str):
        val = int(param)
        if val < 0:
            raise ArgumentTypeError("Expected integer greater than or equal to zero.")
        return val

    args.add_argument("-s", "--skip", type=int_gte_zero, required=False, default=0,
                      help="Skip to the given experiment number")

    args = vars(args.parse_args())
    QUIET = args["quiet"]
    SKIP = args["skip"]

    GRAPHS = [
        'twointersect',
        'crazy',
        'lots_of_triangles',
        'not_a_star',
        'quad_n_tri',
    ]

    MODES = [
        'way',
        'graph',
    ]

    TEST_CASES = [
        'no_simp',
        'los_simp',
        'tri_simp',
    ]

    THICKNESSES = [i for i in range(2, 16)]

    try:
        os.makedirs(str(WORKING_DIRECTORY / 'experiments'))
    except:
        pass

    with (WORKING_DIRECTORY / 'experiments' / time.strftime("experiments_%Y-%m-%d_%H-%M-%S.csv")).open('w') as out:
        out.write("Num,Graph,Mode,TestCase,Thickness,TimeElapsed,Correct,Missing,Extra,Nodes,Edges\n")
        out.flush()


        def do_experiment(num, graph, mode, test_case, thickness):
            qprint("Conducting Experiment #%d" % num)
            elapsed, correct, missing, extra, nodes, edges = -1, -1, -1, -1, -1, -1
            try:
                elapsed, correct, missing, extra, nodes, edges = run_comparer(
                    graph_path="./paths/%s.txt" % graph,
                    mode=mode,
                    test_case=test_case,
                    thickness=thickness,
                )
            except:
                traceback.print_exc()
                qprint("Experiment failed.")

            out.write("%d,%s,%s,%s,%d,%f,%d,%d,%d,%d,%d\n" % (num, graph, mode, test_case, thickness, elapsed, correct, missing, extra, nodes, edges))
            out.flush()

        num = -1
        toskip = SKIP
        if toskip > 0:
            qprint("Skipping to experiment #%d" % toskip)
        for graph in GRAPHS:
            for mode in MODES:
                for test_case in TEST_CASES:
                    for thickness in THICKNESSES:
                        num += 1
                        if toskip > 0:
                            toskip -= 1
                            continue
                        do_experiment(num, graph, mode, test_case, thickness)
