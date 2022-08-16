import sys

sys.path.append(r"/usr/local/lib/python3.6/dist-packages")

import cplex
from cplex.callbacks import UserCutCallback, LazyConstraintCallback