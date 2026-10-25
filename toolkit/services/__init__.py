"""Services layer (v5.1 step 1.5d): what the tools do, with no user interface and no I/O.

Training, evaluation and prediction (``ids``), scanning logic and lookups (``recon``), the
web trainer's workspaces and jobs (``web``). Services use ``toolkit.data`` for files and
the network; they never print, ask, or use Flask or argparse.
"""
