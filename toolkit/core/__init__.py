"""What both tools share (v5.2 step 2.3): the services (``core.services``: training,
evaluation and prediction for the IDS, the scan and its lookups for recon) and the data
they read and write (``core.data``). No user interface: ``toolkit.cli`` and
``toolkit.web`` use this package, and it never imports them."""
