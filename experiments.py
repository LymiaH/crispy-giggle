import subprocess
import tempfile
import traceback
from argparse import ArgumentParser
from pathlib import Path

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
    :return: (CORRECT, MISSING, EXTRA)
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
    p = subprocess.Popen(
        args,
        cwd=str(WORKING_DIRECTORY),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines = True,
        text=True,
        shell=True
    )
    stdout, stderr = p.communicate()  # type: str, str
    qprint("[EXPERIMENT] Result: " + stdout)
    if stderr and len(stderr) > 0:
        qprint("[EXPERIMENT] Errors: " + stderr)
    return tuple(int(num) for num in stdout.strip().split(','))


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("-q", "--quiet", action='store_true', required=False,
                      help="Don't print information to stderr.")

    args = vars(args.parse_args())
    QUIET = args["quiet"]

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

    with open('experiments.csv', 'w') as out:
        out.write("Graph,Mode,TestCase,Thickness,Correct,Missing,Extra\n")
        out.flush()


        def do_experiment(graph, mode, test_case, thickness):
            correct, missing, extra = 0, 0, 0
            try:
                correct, missing, extra = run_comparer(
                    graph_path="./paths/%s.txt" % graph,
                    mode=mode,
                    test_case=test_case,
                    thickness=thickness,
                )
            except:
                traceback.print_exc()
                qprint("Experiment failed.")

            out.write("%s,%s,%s,%d,%d,%d,%d\n" % (graph, mode, test_case, thickness, correct, missing, extra))
            out.flush()


        for graph in GRAPHS:
            for mode in MODES:
                for test_case in TEST_CASES:
                    for thickness in THICKNESSES:
                        do_experiment(graph, mode, test_case, thickness)
