"""Services shared by both tools (v5.1 step 1.5d, in core since v5.2 step 2.3b): what the
tools do, with no user interface and no I/O. Training, evaluation and prediction
(``ids``), scanning logic and lookups (``recon``), the interfaces between layers
(``ports``). They use ``toolkit.core.data`` for files and the network; they never print,
ask, or use Flask or argparse.
"""
